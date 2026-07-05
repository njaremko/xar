#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rust_2018_idioms)]

//! A segmented exponential array with stable element addresses.
//!
//! `xar` stores elements in chunks mapped by power-of-two index ranges: the
//! first two chunks have `1 << BASE_SHIFT` slots each, then each later chunk
//! doubles. Chunks are allocated independently and are never reallocated, so
//! the address of an initialized element does not change when later elements
//! are pushed.
//!
//! This is not a drop-in replacement for [`Vec`]. It is intentionally not
//! contiguous. Use [`chunks`](ExponentialArray::chunks) or
//! [`chunks_mut`](ExponentialArray::chunks_mut) when an API needs contiguous
//! slices.
//!
//! # Examples
//!
//! Basic indexed use:
//!
//! ```
//! use xar::Xar;
//!
//! let mut xs = Xar::new();
//! let first = xs.push("first");
//! let second = xs.push("second");
//!
//! assert_eq!(first, 0);
//! assert_eq!(second, 1);
//! assert_eq!(xs[0], "first");
//! assert_eq!(xs[1], "second");
//! ```
//!
//! Stable raw pointers:
//!
//! ```
//! use xar::Xar;
//!
//! let mut xs = Xar::new();
//! let root = xs.push_ptr(String::from("root"));
//!
//! for i in 0..10_000 {
//!     xs.push(i.to_string());
//! }
//!
//! // The pointer remains non-dangling because the root element was not
//! // removed and `xs` is still alive. Dereferencing a raw pointer is still
//! // unsafe because Rust cannot prove aliasing rules for the caller.
//! assert_eq!(unsafe { root.as_ref() }, "root");
//! ```
//!
//! Iterating by contiguous chunk:
//!
//! ```
//! use xar::ExponentialArray;
//!
//! let xs = (0..10).collect::<ExponentialArray<_, 2, 4>>();
//! let chunk_lengths = xs.chunks().map(<[i32]>::len).collect::<Vec<_>>();
//!
//! assert_eq!(chunk_lengths, vec![4, 4, 2]);
//! ```

extern crate alloc;

use alloc::alloc::{alloc, dealloc, handle_alloc_error};
use alloc::vec::Vec;
use core::alloc::Layout;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::{Extend, FromIterator, FusedIterator};
use core::marker::PhantomData;
use core::mem::{self, MaybeUninit};
use core::ops::{Bound, Index, IndexMut, RangeBounds};
use core::ptr::{self, NonNull};
use core::slice;

#[cfg(test)]
extern crate std;

/// The default base-2 exponent for the first chunk.
///
/// With the default value, the first two chunks hold 16 elements each, the
/// third holds 32, the fourth holds 64, and so on.
pub const DEFAULT_BASE_SHIFT: usize = 4;

/// The default number of chunk pointers stored inline on 64-bit platforms.
#[cfg(target_pointer_width = "64")]
pub const DEFAULT_CHUNKS: usize = 32;

/// The default number of chunk pointers stored inline on 32-bit platforms.
#[cfg(target_pointer_width = "32")]
pub const DEFAULT_CHUNKS: usize = 28;

/// The default number of chunk pointers stored inline on 16-bit platforms.
#[cfg(target_pointer_width = "16")]
pub const DEFAULT_CHUNKS: usize = 12;

/// The default number of chunk pointers stored inline on unusual pointer-width platforms.
#[cfg(not(any(
    target_pointer_width = "16",
    target_pointer_width = "32",
    target_pointer_width = "64"
)))]
pub const DEFAULT_CHUNKS: usize = 16;

/// A default exponential array.
///
/// This alias uses [`DEFAULT_BASE_SHIFT`] and [`DEFAULT_CHUNKS`]. Use
/// [`ExponentialArray`] directly when a different first chunk size or maximum
/// number of chunks is required.
pub type Xar<T> = ExponentialArray<T, DEFAULT_BASE_SHIFT, DEFAULT_CHUNKS>;

/// A segmented exponential array.
///
/// Chunk `0` stores indices below `1 << BASE_SHIFT`. Chunk `c > 0` stores the
/// power-of-two range from `1 << (BASE_SHIFT + c - 1)` up to, but not
/// including, `1 << (BASE_SHIFT + c)`. `CHUNKS` is the fixed number of inline
/// chunk pointers in the container metadata.
///
/// # Address stability
///
/// For non-zero-sized `T`, the address of an element does not change while the
/// element remains initialized in the array. `push`, `push_mut`, `push_ptr`,
/// `reserve`, and `try_reserve` do not move existing elements.
///
/// An element stops being initialized when it is removed by [`pop`](Self::pop),
/// [`truncate`](Self::truncate), [`clear`](Self::clear), a shrinking
/// [`resize`](Self::resize) or [`resize_with`](Self::resize_with), when it is
/// moved out of the source array by [`append`](Self::append), when it is in the
/// tail moved out by [`split_off`](Self::split_off), or when the whole array is
/// dropped. Any raw pointer to that element must then be treated as invalid.
/// For zero-sized types, addresses are not meaningful.
///
/// # Contiguity
///
/// The whole array is not contiguous. Each individual chunk is contiguous.
pub struct ExponentialArray<T, const BASE_SHIFT: usize, const CHUNKS: usize> {
    len: usize,
    // Saturating sum of capacities for the allocated prefix
    // `chunks[..allocated_chunks]`.
    capacity: usize,
    allocated_chunks: usize,
    // Cursor for the next append slot; equivalent to `end_position(len)`.
    tail_chunk: usize,
    tail_offset: usize,
    chunks: [Option<NonNull<MaybeUninit<T>>>; CHUNKS],
}

// SAFETY: the array owns its element storage. Sending ownership to another
// thread is sound when owned elements can be sent.
unsafe impl<T: Send, const BASE_SHIFT: usize, const CHUNKS: usize> Send
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
}

// SAFETY: shared access only yields shared access to elements. This is sound
// when shared references to elements are thread-safe.
unsafe impl<T: Sync, const BASE_SHIFT: usize, const CHUNKS: usize> Sync
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
}

/// The reason a reservation failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TryReserveErrorKind {
    /// The requested element count overflowed `usize`.
    CapacityOverflow,

    /// The requested element count is larger than this `ExponentialArray`
    /// configuration can represent.
    CapacityExceeded {
        /// The requested number of elements.
        requested: usize,
        /// The maximum number of elements this configuration can hold for `T`.
        max: usize,
    },

    /// The allocator returned null for the requested layout.
    AllocError {
        /// The requested allocation size in bytes.
        size: usize,
        /// The requested allocation alignment in bytes.
        align: usize,
    },
}

/// An error returned by fallible reservation APIs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TryReserveError {
    kind: TryReserveErrorKind,
}

impl TryReserveError {
    /// Returns the error kind.
    #[must_use]
    pub const fn kind(&self) -> TryReserveErrorKind {
        self.kind
    }

    const fn capacity_overflow() -> Self {
        Self {
            kind: TryReserveErrorKind::CapacityOverflow,
        }
    }

    const fn capacity_exceeded(requested: usize, max: usize) -> Self {
        Self {
            kind: TryReserveErrorKind::CapacityExceeded { requested, max },
        }
    }

    fn alloc_error(layout: Layout) -> Self {
        Self {
            kind: TryReserveErrorKind::AllocError {
                size: layout.size(),
                align: layout.align(),
            },
        }
    }
}

impl fmt::Display for TryReserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            TryReserveErrorKind::CapacityOverflow => {
                f.write_str("requested capacity overflows usize")
            }
            TryReserveErrorKind::CapacityExceeded { requested, max } => write!(
                f,
                "requested capacity {requested} exceeds maximum capacity {max}"
            ),
            TryReserveErrorKind::AllocError { size, align } => write!(
                f,
                "memory allocation failed for layout with size {size} and align {align}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TryReserveError {}

/// An error returned by [`ExponentialArray::try_push`].
pub struct TryPushError<T> {
    value: T,
    error: TryReserveError,
}

impl<T> TryPushError<T> {
    /// Returns the value that could not be pushed.
    #[must_use]
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Returns the reservation error.
    #[must_use]
    pub const fn error(&self) -> TryReserveError {
        self.error
    }

    /// Splits the error into the original value and reservation error.
    #[must_use]
    pub fn into_parts(self) -> (T, TryReserveError) {
        (self.value, self.error)
    }
}

impl<T> fmt::Debug for TryPushError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TryPushError")
            .field("value", &"<value>")
            .field("error", &self.error)
            .finish()
    }
}

impl<T> fmt::Display for TryPushError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "could not push value: {}", self.error)
    }
}

#[cfg(feature = "std")]
impl<T> std::error::Error for TryPushError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> ExponentialArray<T, BASE_SHIFT, CHUNKS> {
    /// Creates an empty array.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            len: 0,
            capacity: 0,
            allocated_chunks: 0,
            tail_chunk: 0,
            tail_offset: 0,
            chunks: [None; CHUNKS],
        }
    }

    /// Creates an empty array with enough chunks allocated for `capacity`
    /// elements.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` exceeds [`max_capacity`](Self::max_capacity). On
    /// allocation failure, this uses the global allocation error handler.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let mut array = Self::new();
        array.reserve(capacity);
        array
    }

    /// Creates an empty array with enough chunks allocated for `capacity`
    /// elements.
    pub fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        let mut array = Self::new();
        array.try_reserve(capacity)?;
        Ok(array)
    }

    /// Returns the number of initialized elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the array is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a shared reference to the first element, or `None` if empty.
    #[must_use]
    pub fn first(&self) -> Option<&T> {
        self.get(0)
    }

    /// Returns a mutable reference to the first element, or `None` if empty.
    #[must_use]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.get_mut(0)
    }

    /// Returns a shared reference to the last element, or `None` if empty.
    #[must_use]
    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            self.get(self.len - 1)
        }
    }

    /// Returns a mutable reference to the last element, or `None` if empty.
    #[must_use]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            None
        } else {
            self.get_mut(self.len - 1)
        }
    }

    /// Returns a stable raw pointer to the last element, or `None` if empty.
    ///
    /// The pointer remains non-dangling until the element is removed or the
    /// array is dropped. Dereferencing the pointer is unsafe and must obey
    /// Rust's aliasing rules.
    #[must_use]
    pub fn last_ptr(&self) -> Option<NonNull<T>> {
        if self.len == 0 {
            None
        } else {
            self.ptr(self.len - 1)
        }
    }

    fn debug_assert_invariants(&self) {
        if !cfg!(debug_assertions) {
            return;
        }

        debug_assert!(self.len <= self.capacity);
        debug_assert!(self.allocated_chunks <= CHUNKS);

        let (tail_chunk, tail_offset) = Self::end_position(self.len);
        debug_assert_eq!(self.tail_chunk, tail_chunk);
        debug_assert_eq!(self.tail_offset, tail_offset);

        let mut expected_capacity = 0usize;
        let mut chunk = 0usize;
        while chunk < self.allocated_chunks {
            debug_assert!(self.chunks[chunk].is_some());
            let capacity = Self::chunk_capacity(chunk).expect("allocated chunk is representable");
            expected_capacity = expected_capacity.saturating_add(capacity);
            chunk += 1;
        }
        debug_assert_eq!(self.capacity, expected_capacity);

        while chunk < CHUNKS {
            debug_assert!(self.chunks[chunk].is_none());
            chunk += 1;
        }
    }

    /// Returns the number of elements that can be held without allocating more
    /// chunks.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the maximum number of elements this configuration can hold for
    /// `T`.
    ///
    /// This is the sum of the representable chunk capacities, capped by Rust's
    /// allocation layout limits for `T`.
    #[must_use]
    pub fn max_capacity() -> usize {
        let mut total = 0usize;
        let mut chunk = 0usize;

        while chunk < CHUNKS {
            let Some(capacity) = Self::chunk_capacity(chunk) else {
                break;
            };

            if mem::size_of::<T>() != 0 && Layout::array::<MaybeUninit<T>>(capacity).is_err() {
                break;
            }

            let Some(next) = total.checked_add(capacity) else {
                return usize::MAX;
            };
            total = next;
            chunk += 1;
        }

        total
    }

    /// Returns the number of chunks currently allocated.
    #[must_use]
    pub const fn allocated_chunks(&self) -> usize {
        self.allocated_chunks
    }

    /// Reserves capacity for at least `additional` more elements.
    ///
    /// This may allocate one or more chunks. Existing elements are not moved.
    ///
    /// # Panics
    ///
    /// Panics if the requested capacity exceeds [`max_capacity`](Self::max_capacity).
    /// On allocation failure, this uses the global allocation error handler.
    pub fn reserve(&mut self, additional: usize) {
        if let Err(error) = self.try_reserve(additional) {
            panic_or_handle_reserve(error);
        }
    }

    /// Tries to reserve capacity for at least `additional` more elements.
    ///
    /// Existing elements are not moved. If this returns an error, `self.len()` is
    /// unchanged. Capacity may have increased if allocation of an earlier chunk
    /// succeeded before a later chunk failed.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let requested_capacity = self
            .len
            .checked_add(additional)
            .ok_or_else(TryReserveError::capacity_overflow)?;

        self.debug_assert_invariants();

        if requested_capacity <= self.capacity {
            return Ok(());
        }

        let result = self.try_reserve_slow(requested_capacity);
        if result.is_ok() {
            self.debug_assert_invariants();
        }
        result
    }

    #[cold]
    #[inline(never)]
    fn try_reserve_slow(&mut self, requested_capacity: usize) -> Result<(), TryReserveError> {
        let max = Self::max_capacity();
        if requested_capacity > max {
            return Err(TryReserveError::capacity_exceeded(requested_capacity, max));
        }

        if requested_capacity == 0 {
            return Ok(());
        }

        let (last_chunk, _) = Self::locate(requested_capacity - 1);
        while self.allocated_chunks <= last_chunk {
            self.try_allocate_next_chunk()?;
        }

        Ok(())
    }

    /// Appends `value` and returns its index.
    ///
    /// Existing elements are not moved.
    ///
    /// # Panics
    ///
    /// Panics if the array is full. On allocation failure, this uses the global
    /// allocation error handler.
    pub fn push(&mut self, value: T) -> usize {
        match self.try_push(value) {
            Ok(index) => index,
            Err(error) => {
                let (_, reserve_error) = error.into_parts();
                panic_or_handle_reserve(reserve_error);
            }
        }
    }

    /// Appends a value produced by `make_value` and returns its index.
    ///
    /// The closure is not called if reserving space fails.
    ///
    /// # Panics
    ///
    /// Panics if the array is full. On allocation failure, this uses the global
    /// allocation error handler.
    pub fn push_with<F>(&mut self, make_value: F) -> usize
    where
        F: FnOnce() -> T,
    {
        match self.try_push_with(make_value) {
            Ok(index) => index,
            Err(error) => panic_or_handle_reserve(error),
        }
    }

    /// Appends `value` and returns a mutable reference to it.
    ///
    /// Existing elements are not moved.
    ///
    /// # Panics
    ///
    /// Panics if the array is full. On allocation failure, this uses the global
    /// allocation error handler.
    pub fn push_mut(&mut self, value: T) -> &mut T {
        let (index, chunk, offset) = match self.reserve_tail_slot() {
            Ok(slot) => slot,
            Err(error) => panic_or_handle_reserve(error),
        };
        // SAFETY: reservation above guarantees storage for `index`, and the
        // returned pointer names the initialized element.
        unsafe { &mut *self.write_reserved_tail_slot(index, chunk, offset, value) }
    }

    /// Appends `value` and returns a stable raw pointer to it.
    ///
    /// The pointer remains non-dangling until the element is removed or the
    /// array is dropped. Dereferencing the pointer is unsafe and must obey
    /// Rust's aliasing rules.
    ///
    /// # Panics
    ///
    /// Panics if the array is full. On allocation failure, this uses the global
    /// allocation error handler.
    pub fn push_ptr(&mut self, value: T) -> NonNull<T> {
        let (index, chunk, offset) = match self.reserve_tail_slot() {
            Ok(slot) => slot,
            Err(error) => panic_or_handle_reserve(error),
        };
        // SAFETY: reservation above guarantees storage for `index`, and the
        // returned pointer is non-null.
        unsafe {
            NonNull::new_unchecked(self.write_reserved_tail_slot(index, chunk, offset, value))
        }
    }

    /// Tries to append `value` and returns its index.
    pub fn try_push(&mut self, value: T) -> Result<usize, TryPushError<T>> {
        let (index, chunk, offset) = match self.reserve_tail_slot() {
            Ok(slot) => slot,
            Err(error) => {
                return Err(TryPushError { value, error });
            }
        };

        // SAFETY: reservation above guarantees storage for `index`.
        unsafe { self.write_reserved_tail_slot(index, chunk, offset, value) };
        Ok(index)
    }

    /// Appends clones of all elements in `values`.
    ///
    /// Existing elements are not moved. If reserving space fails, no elements
    /// are appended.
    ///
    /// # Panics
    ///
    /// Panics if the requested capacity exceeds [`max_capacity`](Self::max_capacity).
    /// On allocation failure, this uses the global allocation error handler.
    pub fn extend_from_slice(&mut self, values: &[T])
    where
        T: Clone,
    {
        self.reserve(values.len());
        // SAFETY: the reservation above guarantees enough uninitialized tail
        // slots for all cloned values.
        unsafe { self.clone_from_ptr_to_tail_reserved(values.as_ptr(), values.len()) };
    }

    /// Appends clones of the elements in `range` to the end of the array.
    ///
    /// Existing elements are not moved. The source range is resolved before any
    /// new elements are appended, matching [`Vec::extend_from_within`].
    ///
    /// # Panics
    ///
    /// Panics if `range` is out of bounds, if the requested capacity exceeds
    /// [`max_capacity`](Self::max_capacity), or if cloning an element panics. On
    /// allocation failure, this uses the global allocation error handler.
    pub fn extend_from_within<R>(&mut self, range: R)
    where
        T: Clone,
        R: RangeBounds<usize>,
    {
        let (start, end) = self.resolve_range_bounds(range);
        let additional = end - start;
        if additional == 0 {
            return;
        }

        self.reserve(additional);
        // SAFETY: `start..end` was resolved against the old initialized length,
        // and reservation above guarantees enough disjoint tail slots for the
        // cloned values.
        unsafe { self.clone_array_range_to_tail_reserved(start, end) };
    }

    /// Moves all elements from `other` to the end of `self`, leaving `other`
    /// empty.
    ///
    /// Existing elements in `self` are not moved. Raw pointers to elements that
    /// were in `other` become invalid because those elements are removed from
    /// `other` before being appended to `self`.
    ///
    /// # Panics
    ///
    /// Panics if the requested capacity exceeds [`max_capacity`](Self::max_capacity).
    /// On allocation failure, this uses the global allocation error handler. If
    /// reserving space fails, both arrays keep their original elements.
    pub fn append(&mut self, other: &mut Self) {
        let count = other.len;
        if count == 0 {
            return;
        }

        self.reserve(count);

        let other_array = other as *mut Self;
        // SAFETY: `other_array` owns initialized elements in `0..count`, and
        // reservation above guarantees disjoint uninitialized tail slots in
        // `self`. After the non-panicking bulk moves, `other` is marked empty so
        // those moved values are not dropped twice.
        unsafe { self.move_array_range_to_tail_unchecked(other_array.cast_const(), 0, count) };
        other.len = 0;
        other.tail_chunk = 0;
        other.tail_offset = 0;
        other.debug_assert_invariants();
    }

    unsafe fn move_array_range_to_tail_unchecked(
        &mut self,
        source: *const Self,
        start: usize,
        end: usize,
    ) {
        debug_assert!(start <= end);

        let mut index = start;
        while index < end {
            let (chunk, offset) = Self::locate(index);
            let chunk_end = Self::chunk_start_unchecked(chunk)
                .saturating_add(Self::chunk_capacity_unchecked(chunk))
                .min(end);
            let count = chunk_end - index;
            // SAFETY: caller guarantees `source[index..end]` is initialized;
            // this subrange lies in one allocated chunk.
            let source_ptr = unsafe { Self::ptr_at_chunk_offset_unchecked(source, chunk, offset) };
            // SAFETY: caller guarantees enough reserved tail capacity and that
            // source and destination ranges do not overlap.
            unsafe { self.move_from_ptr_to_tail_unchecked(source_ptr, count) };
            index = chunk_end;
        }
    }

    unsafe fn move_array_range_to_ptr_unchecked(
        source: *const Self,
        start: usize,
        end: usize,
        mut target: *mut T,
    ) {
        debug_assert!(start <= end);

        let mut index = start;
        while index < end {
            let (chunk, offset) = Self::locate(index);
            let chunk_end = Self::chunk_start_unchecked(chunk)
                .saturating_add(Self::chunk_capacity_unchecked(chunk))
                .min(end);
            let count = chunk_end - index;
            // SAFETY: caller guarantees `source[index..end]` is initialized;
            // this subrange lies in one allocated chunk.
            let source_ptr = unsafe { Self::ptr_at_chunk_offset_unchecked(source, chunk, offset) };
            if mem::size_of::<T>() != 0 {
                // SAFETY: caller guarantees `target..target + count` is
                // uninitialized writable storage and does not overlap source.
                unsafe { ptr::copy_nonoverlapping(source_ptr, target, count) };
                // SAFETY: the destination range just written has `count`
                // elements.
                target = unsafe { target.add(count) };
            }
            index = chunk_end;
        }
    }

    unsafe fn move_from_ptr_to_tail_unchecked(&mut self, mut source: *const T, mut count: usize) {
        debug_assert!(self.len.checked_add(count).is_some());
        debug_assert!(self.len + count <= self.capacity);

        if count == 0 {
            return;
        }

        if mem::size_of::<T>() == 0 {
            self.advance_tail_after_bulk_write(count);
            return;
        }

        while count != 0 {
            debug_assert!(self.tail_chunk < CHUNKS);
            let chunk_capacity = Self::chunk_capacity_unchecked(self.tail_chunk);
            let chunk_remaining = chunk_capacity - self.tail_offset;
            let write_count = count.min(chunk_remaining);
            // SAFETY: caller guarantees the tail range is reserved and
            // uninitialized; this subrange lies in one allocated chunk.
            let target = unsafe {
                Self::ptr_at_chunk_offset_unchecked(self, self.tail_chunk, self.tail_offset)
            };
            // SAFETY: caller guarantees source and destination are initialized
            // and uninitialized respectively, and do not overlap.
            unsafe { ptr::copy_nonoverlapping(source, target, write_count) };
            self.len += write_count;
            self.tail_offset += write_count;
            if self.tail_offset == chunk_capacity {
                self.tail_chunk += 1;
                self.tail_offset = 0;
            }
            // SAFETY: the just-moved source subrange had `write_count`
            // elements.
            source = unsafe { source.add(write_count) };
            count -= write_count;
        }
    }

    unsafe fn clone_array_range_to_tail_reserved(&mut self, start: usize, end: usize)
    where
        T: Clone,
    {
        debug_assert!(start <= end);
        debug_assert!(end <= self.len);

        let array = self as *const Self;
        let mut index = start;
        while index < end {
            let (chunk, offset) = Self::locate(index);
            let chunk_end = Self::chunk_start_unchecked(chunk)
                .saturating_add(Self::chunk_capacity_unchecked(chunk))
                .min(end);
            let count = chunk_end - index;
            // SAFETY: `start..end` is initialized and this subrange lies in one
            // allocated chunk. The destination tail is disjoint because the
            // range was resolved before appending.
            let source = unsafe { Self::ptr_at_chunk_offset_unchecked(array, chunk, offset) };
            // SAFETY: caller reserved enough tail capacity.
            unsafe { self.clone_from_ptr_to_tail_reserved(source, count) };
            index = chunk_end;
        }
    }

    unsafe fn clone_from_ptr_to_tail_reserved(&mut self, mut source: *const T, mut count: usize)
    where
        T: Clone,
    {
        debug_assert!(self.len.checked_add(count).is_some());
        debug_assert!(self.len + count <= self.capacity);

        if count == 0 {
            return;
        }

        let mut guard = TailInitGuard::new(self);
        let mut tail_chunk = self.tail_chunk;
        let mut tail_offset = self.tail_offset;
        while count != 0 {
            debug_assert!(tail_chunk < CHUNKS);
            let chunk_capacity = Self::chunk_capacity_unchecked(tail_chunk);
            let chunk_remaining = chunk_capacity - tail_offset;
            let write_count = count.min(chunk_remaining);
            // SAFETY: caller reserved the tail slots, and this subrange lies in
            // one allocated chunk.
            let target =
                unsafe { Self::ptr_at_chunk_offset_unchecked(self, tail_chunk, tail_offset) };

            let mut offset = 0usize;
            while offset < write_count {
                // SAFETY: caller guarantees `source..source + count` names
                // initialized elements. It may point inside `self`, but never
                // into the reserved tail range currently being written.
                let value = unsafe { (&*source.add(offset)).clone() };
                // SAFETY: `target + offset` is an uninitialized reserved slot.
                unsafe { target.add(offset).write(value) };
                guard.initialized += 1;
                offset += 1;
            }

            tail_offset += write_count;
            if tail_offset == chunk_capacity {
                tail_chunk += 1;
                tail_offset = 0;
            }
            // SAFETY: the source subrange just cloned had `write_count`
            // elements.
            source = unsafe { source.add(write_count) };
            count -= write_count;
        }
        guard.commit();
    }

    unsafe fn fill_tail_reserved_with<F>(&mut self, mut count: usize, mut make_value: F)
    where
        F: FnMut() -> T,
    {
        debug_assert!(self.len.checked_add(count).is_some());
        debug_assert!(self.len + count <= self.capacity);

        if count == 0 {
            return;
        }

        let mut guard = TailInitGuard::new(self);
        let mut tail_chunk = self.tail_chunk;
        let mut tail_offset = self.tail_offset;
        while count != 0 {
            debug_assert!(tail_chunk < CHUNKS);
            let chunk_capacity = Self::chunk_capacity_unchecked(tail_chunk);
            let chunk_remaining = chunk_capacity - tail_offset;
            let write_count = count.min(chunk_remaining);
            // SAFETY: caller reserved the tail slots, and this subrange lies in
            // one allocated chunk.
            let target =
                unsafe { Self::ptr_at_chunk_offset_unchecked(self, tail_chunk, tail_offset) };

            let mut offset = 0usize;
            while offset < write_count {
                let value = make_value();
                // SAFETY: `target + offset` is an uninitialized reserved slot.
                unsafe { target.add(offset).write(value) };
                guard.initialized += 1;
                offset += 1;
            }

            tail_offset += write_count;
            if tail_offset == chunk_capacity {
                tail_chunk += 1;
                tail_offset = 0;
            }
            count -= write_count;
        }
        guard.commit();
    }

    unsafe fn copy_refs_to_tail_reserved<'a, I>(&mut self, iterator: &mut I, mut count: usize)
    where
        T: Copy + 'a,
        I: Iterator<Item = &'a T>,
    {
        debug_assert!(self.len.checked_add(count).is_some());
        debug_assert!(self.len + count <= self.capacity);

        if count == 0 {
            return;
        }

        let mut guard = TailInitGuard::new(self);
        let mut tail_chunk = self.tail_chunk;
        let mut tail_offset = self.tail_offset;
        while count != 0 {
            debug_assert!(tail_chunk < CHUNKS);
            let chunk_capacity = Self::chunk_capacity_unchecked(tail_chunk);
            let chunk_remaining = chunk_capacity - tail_offset;
            let write_count = count.min(chunk_remaining);
            // SAFETY: caller reserved the tail slots, and this subrange lies in
            // one allocated chunk.
            let target =
                unsafe { Self::ptr_at_chunk_offset_unchecked(self, tail_chunk, tail_offset) };

            let mut offset = 0usize;
            while offset < write_count {
                let Some(value) = iterator.next() else {
                    guard.commit();
                    return;
                };
                // SAFETY: `target + offset` is an uninitialized reserved slot.
                unsafe { target.add(offset).write(*value) };
                guard.initialized += 1;
                offset += 1;
            }

            tail_offset += write_count;
            if tail_offset == chunk_capacity {
                tail_chunk += 1;
                tail_offset = 0;
            }
            count -= write_count;
        }
        guard.commit();
    }

    fn advance_tail_after_bulk_write(&mut self, count: usize) {
        debug_assert!(self.len.checked_add(count).is_some());
        debug_assert!(self.len + count <= self.capacity);

        if count == 0 {
            return;
        }

        self.len += count;
        let (tail_chunk, tail_offset) = Self::end_position(self.len);
        self.tail_chunk = tail_chunk;
        self.tail_offset = tail_offset;
        self.debug_assert_invariants();
    }

    fn reserve_tail_slot(&mut self) -> Result<(usize, usize, usize), TryReserveError> {
        let index = self.len;
        self.try_reserve(1)?;

        Ok((index, self.tail_chunk, self.tail_offset))
    }

    /// Tries to append a value produced by `make_value` and returns its index.
    ///
    /// The closure is not called if reserving space fails.
    pub fn try_push_with<F>(&mut self, make_value: F) -> Result<usize, TryReserveError>
    where
        F: FnOnce() -> T,
    {
        let (index, chunk, offset) = self.reserve_tail_slot()?;
        let value = make_value();

        // SAFETY: reservation above guarantees storage for `index`.
        unsafe { self.write_reserved_tail_slot(index, chunk, offset, value) };
        Ok(index)
    }

    /// Removes the last element and returns it, or returns `None` if the array
    /// is empty.
    ///
    /// Allocated chunks are retained.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        let index = self.len - 1;
        let (chunk, offset) = self.retreat_tail_before_pop();
        self.len = index;
        self.debug_assert_invariants();
        // SAFETY: the old last element was initialized, and `len` has been
        // reduced so it will not be dropped again.
        Some(unsafe { Self::ptr_at_chunk_offset_unchecked(self, chunk, offset).read() })
    }

    /// Shortens the array, keeping the first `new_len` elements.
    ///
    /// If `new_len` is greater than or equal to the current length, this does
    /// nothing. Allocated chunks are retained.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }

        let old_len = self.len;
        let (tail_chunk, tail_offset) = Self::end_position(new_len);
        self.len = new_len;
        self.tail_chunk = tail_chunk;
        self.tail_offset = tail_offset;
        self.debug_assert_invariants();
        // SAFETY: `new_len..old_len` contains initialized elements, and `len`
        // has already been shortened so a panic while dropping cannot drop the
        // same elements again.
        unsafe { self.drop_range_unchecked(new_len, old_len) };
    }

    /// Resizes the array to `new_len`.
    ///
    /// If `new_len` is less than the current length, the array is truncated. If
    /// it is greater, clones of `value` are appended until the requested length
    /// is reached. Existing elements that remain in the array are not moved.
    ///
    /// # Panics
    ///
    /// Panics if the requested capacity exceeds [`max_capacity`](Self::max_capacity),
    /// or if cloning `value` panics. On allocation failure, this uses the global
    /// allocation error handler.
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        let len = self.len;
        if new_len <= len {
            self.truncate(new_len);
            return;
        }

        let additional = new_len - len;
        self.reserve(additional);

        if additional == 1 {
            self.push(value);
            return;
        }

        // SAFETY: the reservation above guarantees enough uninitialized tail
        // slots for all appended clones and the final moved value.
        unsafe {
            let value_ptr = &value as *const T;
            self.fill_tail_reserved_with(additional - 1, || (&*value_ptr).clone());
            self.move_from_ptr_to_tail_unchecked(value_ptr, 1);
        }
        mem::forget(value);
    }

    /// Resizes the array to `new_len` using `make_value` for appended elements.
    ///
    /// If `new_len` is less than the current length, the array is truncated and
    /// `make_value` is not called. If it is greater, `make_value` is called once
    /// for each appended element. Existing elements that remain in the array are
    /// not moved.
    ///
    /// # Panics
    ///
    /// Panics if the requested capacity exceeds [`max_capacity`](Self::max_capacity),
    /// or if `make_value` panics. On allocation failure, this uses the global
    /// allocation error handler.
    pub fn resize_with<F>(&mut self, new_len: usize, make_value: F)
    where
        F: FnMut() -> T,
    {
        let len = self.len;
        if new_len <= len {
            self.truncate(new_len);
            return;
        }

        let additional = new_len - len;
        self.reserve(additional);

        // SAFETY: the reservation above guarantees enough uninitialized tail
        // slots for all generated values.
        unsafe { self.fill_tail_reserved_with(additional, make_value) };
    }

    /// Splits the array into two at `at`.
    ///
    /// `self` keeps `0..at`, and the returned array contains the old
    /// `at..len` tail in order. Existing elements retained by `self` are not
    /// moved. Raw pointers to elements in the split-off tail become invalid
    /// because those elements are removed from `self`.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`, or if allocating the returned array fails. On
    /// allocation failure, this uses the global allocation error handler.
    #[must_use]
    pub fn split_off(&mut self, at: usize) -> Self {
        let len = self.len;
        assert!(
            at <= len,
            "split_off index {at} out of bounds for ExponentialArray with len {len}"
        );

        let tail_len = len - at;
        let mut tail = Self::with_capacity(tail_len);
        if tail_len == 0 {
            return tail;
        }

        let self_array = self as *mut Self;
        // SAFETY: `self_array[at..len]` is initialized, `tail` has enough
        // reserved disjoint storage, and the non-panicking bulk move completes
        // before `self` is shortened so the moved values are not dropped twice.
        unsafe { tail.move_array_range_to_tail_unchecked(self_array.cast_const(), at, len) };
        let (tail_chunk, tail_offset) = Self::end_position(at);
        self.len = at;
        self.tail_chunk = tail_chunk;
        self.tail_offset = tail_offset;
        self.debug_assert_invariants();

        tail
    }

    /// Removes all elements.
    ///
    /// Allocated chunks are retained.
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Returns a shared reference to the element at `index`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            // SAFETY: the bounds check above proves the element is initialized.
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at `index`.
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            // SAFETY: the bounds check above proves the element is initialized.
            Some(unsafe { self.get_unchecked_mut(index) })
        } else {
            None
        }
    }

    /// Returns a shared reference to the element at `index` without bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// `index` must be less than `self.len()`.
    #[must_use]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        debug_assert!(index < self.len);
        // SAFETY: guaranteed by the caller.
        unsafe { &*Self::ptr_at_unchecked_raw(self, index) }
    }

    /// Returns a mutable reference to the element at `index` without bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// `index` must be less than `self.len()`. The caller must also ensure no
    /// other reference to the same element is live for the returned lifetime.
    #[must_use]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        debug_assert!(index < self.len);
        // SAFETY: guaranteed by the caller.
        unsafe { &mut *Self::ptr_at_unchecked_raw(self, index) }
    }

    /// Returns a stable raw pointer to the element at `index`.
    ///
    /// The pointer remains non-dangling until the element is removed or the
    /// array is dropped. Dereferencing the pointer is unsafe and must obey
    /// Rust's aliasing rules.
    #[must_use]
    pub fn ptr(&self, index: usize) -> Option<NonNull<T>> {
        if index < self.len {
            // SAFETY: the pointer is created from an initialized element and is
            // non-null, including the dangling sentinel used for ZSTs.
            Some(unsafe { NonNull::new_unchecked(Self::ptr_at_unchecked_raw(self, index)) })
        } else {
            None
        }
    }

    /// Returns an iterator over shared references.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, T, BASE_SHIFT, CHUNKS> {
        Iter {
            array: self,
            cursor: ElementCursor::new::<T, BASE_SHIFT, CHUNKS>(self.len),
            marker: PhantomData,
        }
    }

    /// Returns an iterator over mutable references.
    #[must_use]
    pub fn iter_mut(&mut self) -> IterMut<'_, T, BASE_SHIFT, CHUNKS> {
        IterMut {
            array: self,
            cursor: ElementCursor::new::<T, BASE_SHIFT, CHUNKS>(self.len),
            marker: PhantomData,
        }
    }

    /// Returns an iterator over initialized contiguous chunks.
    ///
    /// Each yielded slice is contiguous. The whole array is not necessarily
    /// contiguous.
    #[must_use]
    pub fn chunks(&self) -> Chunks<'_, T, BASE_SHIFT, CHUNKS> {
        Chunks {
            array: self,
            front: 0,
            back: self.initialized_chunks(),
            marker: PhantomData,
        }
    }

    /// Returns an iterator over initialized mutable contiguous chunks.
    ///
    /// Each yielded slice is contiguous. The whole array is not necessarily
    /// contiguous.
    #[must_use]
    pub fn chunks_mut(&mut self) -> ChunksMut<'_, T, BASE_SHIFT, CHUNKS> {
        let back = self.initialized_chunks();
        ChunksMut {
            array: self,
            front: 0,
            back,
            marker: PhantomData,
        }
    }

    fn initialized_chunks(&self) -> usize {
        if self.len == 0 {
            0
        } else {
            Self::locate(self.len - 1).0 + 1
        }
    }

    fn resolve_range_bounds<R>(&self, range: R) -> (usize, usize)
    where
        R: RangeBounds<usize>,
    {
        let start = match range.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start
                .checked_add(1)
                .expect("range start index overflows usize"),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&end) => end.checked_add(1).expect("range end index overflows usize"),
            Bound::Excluded(&end) => end,
            Bound::Unbounded => self.len,
        };

        assert!(
            start <= end,
            "range start index {start} exceeds range end index {end}"
        );
        assert!(
            end <= self.len,
            "range end index {end} out of bounds for ExponentialArray with len {}",
            self.len
        );

        (start, end)
    }

    fn initial_front_remaining(len: usize) -> usize {
        if len == 0 {
            0
        } else {
            len.min(Self::chunk_capacity_unchecked(0))
        }
    }

    fn end_position(len: usize) -> (usize, usize) {
        if len == 0 {
            return (0, 0);
        }

        let (chunk, offset) = Self::locate(len - 1);
        let end_offset = offset + 1;
        if end_offset == Self::chunk_capacity_unchecked(chunk) {
            (chunk + 1, 0)
        } else {
            (chunk, end_offset)
        }
    }

    fn chunk_capacity(chunk: usize) -> Option<usize> {
        if chunk >= CHUNKS {
            return None;
        }

        let shift = Self::chunk_shift(chunk)?;
        Some(Self::capacity_for_shift(shift))
    }

    fn chunk_shift(chunk: usize) -> Option<usize> {
        let growth_shift = if chunk == 0 { 0 } else { chunk - 1 };
        let shift = BASE_SHIFT.checked_add(growth_shift)?;
        if shift >= usize::BITS as usize {
            return None;
        }

        Some(shift)
    }

    fn capacity_for_shift(shift: usize) -> usize {
        debug_assert!(shift < usize::BITS as usize);
        1usize << shift
    }

    fn chunk_layout(chunk: usize) -> Result<Layout, TryReserveError> {
        let capacity =
            Self::chunk_capacity(chunk).ok_or_else(TryReserveError::capacity_overflow)?;
        Layout::array::<MaybeUninit<T>>(capacity).map_err(|_| TryReserveError::capacity_overflow())
    }

    fn chunk_start_unchecked(chunk: usize) -> usize {
        if chunk == 0 {
            0
        } else {
            let shift = BASE_SHIFT + chunk - 1;
            debug_assert!(shift < usize::BITS as usize);
            Self::capacity_for_shift(shift)
        }
    }

    fn chunk_capacity_unchecked(chunk: usize) -> usize {
        let shift = BASE_SHIFT + chunk.saturating_sub(1);
        debug_assert!(shift < usize::BITS as usize);
        Self::capacity_for_shift(shift)
    }

    fn locate(index: usize) -> (usize, usize) {
        debug_assert!(BASE_SHIFT < usize::BITS as usize);

        let scaled_index = index >> BASE_SHIFT;
        if scaled_index == 0 {
            return (0, index);
        }

        let chunk = usize::BITS as usize - scaled_index.leading_zeros() as usize;
        debug_assert!(chunk < CHUNKS);
        let start = Self::chunk_start_unchecked(chunk);
        let capacity = Self::chunk_capacity_unchecked(chunk);
        debug_assert!(index >= start);
        debug_assert!(index - start < capacity);

        (chunk, index - start)
    }

    fn try_allocate_next_chunk(&mut self) -> Result<(), TryReserveError> {
        let chunk = self.allocated_chunks;
        debug_assert!(chunk < CHUNKS);
        debug_assert!(self.chunks[chunk].is_none());

        let capacity =
            Self::chunk_capacity(chunk).ok_or_else(TryReserveError::capacity_overflow)?;
        let next_capacity = self.capacity.saturating_add(capacity);
        if mem::size_of::<T>() == 0 {
            self.chunks[chunk] = Some(NonNull::dangling());
            self.allocated_chunks = chunk + 1;
            self.capacity = next_capacity;
            return Ok(());
        }

        let layout = Self::chunk_layout(chunk)?;

        // SAFETY: `layout` is a valid non-zero layout for `MaybeUninit<T>`
        // elements. The returned allocation is managed by this array and
        // deallocated with the same layout in `Drop`.
        let raw = unsafe { alloc(layout) };
        let Some(pointer) = NonNull::new(raw.cast::<MaybeUninit<T>>()) else {
            return Err(TryReserveError::alloc_error(layout));
        };

        self.chunks[chunk] = Some(pointer);
        self.allocated_chunks = chunk + 1;
        self.capacity = next_capacity;
        Ok(())
    }

    unsafe fn ptr_at_chunk_offset_unchecked(
        array: *const Self,
        chunk: usize,
        offset: usize,
    ) -> *mut T {
        debug_assert!(chunk < CHUNKS);
        debug_assert!(offset < Self::chunk_capacity_unchecked(chunk));

        if mem::size_of::<T>() == 0 {
            return NonNull::<T>::dangling().as_ptr();
        }

        // SAFETY: the caller guarantees that the chunk is allocated and the
        // offset names an initialized or reserved slot in that chunk.
        let base = unsafe { Self::allocated_chunk_pointer_unchecked(array, chunk) };

        // SAFETY: guaranteed by the caller.
        unsafe { base.add(offset).cast::<T>() }
    }

    unsafe fn write_reserved_tail_slot(
        &mut self,
        index: usize,
        chunk: usize,
        offset: usize,
        value: T,
    ) -> *mut T {
        debug_assert_eq!(index, self.len);
        debug_assert_eq!(chunk, self.tail_chunk);
        debug_assert_eq!(offset, self.tail_offset);

        // SAFETY: the caller reserved the tail slot and passed its cursor.
        let pointer = unsafe { Self::ptr_at_chunk_offset_unchecked(self, chunk, offset) };
        // SAFETY: the reserved tail slot is uninitialized.
        unsafe { pointer.write(value) };
        self.len = index + 1;
        self.advance_tail_after_push();
        self.debug_assert_invariants();
        pointer
    }

    fn advance_tail_after_push(&mut self) {
        debug_assert!(self.tail_chunk < CHUNKS);
        debug_assert!(self.tail_offset < Self::chunk_capacity_unchecked(self.tail_chunk));

        self.tail_offset += 1;
        if self.tail_offset == Self::chunk_capacity_unchecked(self.tail_chunk) {
            self.tail_chunk += 1;
            self.tail_offset = 0;
        }
    }

    fn retreat_tail_before_pop(&mut self) -> (usize, usize) {
        debug_assert!(self.len > 0);
        debug_assert!(self.tail_chunk <= CHUNKS);

        if self.tail_offset == 0 {
            debug_assert!(self.tail_chunk > 0);
            self.tail_chunk -= 1;
            self.tail_offset = Self::chunk_capacity_unchecked(self.tail_chunk);
        }

        self.tail_offset -= 1;
        (self.tail_chunk, self.tail_offset)
    }

    unsafe fn drop_range_unchecked(&mut self, start: usize, end: usize) {
        debug_assert!(start <= end);
        debug_assert!(end <= self.capacity);

        if start == end || !mem::needs_drop::<T>() {
            return;
        }

        let array = self as *mut Self;
        let mut guard = DropRangeGuard { array, start, end };

        while guard.start < guard.end {
            let (chunk, offset) = Self::locate(guard.start);
            let chunk_end = Self::chunk_start_unchecked(chunk)
                .saturating_add(Self::chunk_capacity_unchecked(chunk));
            let range_end = guard.end.min(chunk_end);
            let count = range_end - guard.start;
            let pointer = if mem::size_of::<T>() == 0 {
                NonNull::<T>::dangling().as_ptr()
            } else {
                // SAFETY: the range is initialized and contained in this
                // chunk by construction.
                unsafe { Self::ptr_at_chunk_offset_unchecked(array.cast_const(), chunk, offset) }
            };

            guard.start = range_end;
            // SAFETY: `pointer..pointer + count` names initialized elements in
            // one contiguous chunk. The guard already points at the next chunk,
            // so unwinding cannot strand later chunks.
            unsafe { ptr::drop_in_place(ptr::slice_from_raw_parts_mut(pointer, count)) };
        }
    }

    unsafe fn ptr_at_unchecked_raw(array: *const Self, index: usize) -> *mut T {
        if mem::size_of::<T>() == 0 {
            return NonNull::<T>::dangling().as_ptr();
        }

        let (chunk, offset) = Self::locate(index);
        // SAFETY: callers only request initialized/reserved indices, and all
        // chunks up to that index have been allocated.
        unsafe { Self::ptr_at_chunk_offset_unchecked(array, chunk, offset) }
    }

    unsafe fn chunk_slice_from_raw<'a>(array: *const Self, chunk: usize) -> &'a [T] {
        let len = Self::initialized_len_in_chunk_raw(array, chunk);
        // SAFETY: callers only request initialized chunks, starting at offset
        // zero and spanning the initialized prefix of the chunk.
        unsafe { Self::chunk_slice_range_from_raw(array, chunk, 0, len) }
    }

    unsafe fn chunk_slice_range_from_raw<'a>(
        array: *const Self,
        chunk: usize,
        offset: usize,
        len: usize,
    ) -> &'a [T] {
        debug_assert!(offset <= Self::chunk_capacity_unchecked(chunk));
        debug_assert!(len <= Self::chunk_capacity_unchecked(chunk) - offset);

        let ptr = if mem::size_of::<T>() == 0 {
            NonNull::<T>::dangling().as_ptr()
        } else {
            // SAFETY: callers only request initialized elements inside an
            // allocated chunk.
            unsafe { Self::ptr_at_chunk_offset_unchecked(array, chunk, offset) }
        };

        // SAFETY: `ptr..ptr + len` names initialized elements within one
        // contiguous chunk.
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    unsafe fn chunk_slice_mut_from_raw<'a>(array: *mut Self, chunk: usize) -> &'a mut [T] {
        let len = Self::initialized_len_in_chunk_raw(array, chunk);
        // SAFETY: callers only request initialized chunks, starting at offset
        // zero and spanning the initialized prefix of the chunk.
        unsafe { Self::chunk_slice_range_mut_from_raw(array, chunk, 0, len) }
    }

    unsafe fn chunk_slice_range_mut_from_raw<'a>(
        array: *mut Self,
        chunk: usize,
        offset: usize,
        len: usize,
    ) -> &'a mut [T] {
        debug_assert!(offset <= Self::chunk_capacity_unchecked(chunk));
        debug_assert!(len <= Self::chunk_capacity_unchecked(chunk) - offset);

        let ptr = if mem::size_of::<T>() == 0 {
            NonNull::<T>::dangling().as_ptr()
        } else {
            // SAFETY: callers only request initialized elements inside an
            // allocated chunk.
            unsafe { Self::ptr_at_chunk_offset_unchecked(array.cast_const(), chunk, offset) }
        };

        // SAFETY: `ptr..ptr + len` names initialized elements within one
        // contiguous chunk, and callers guarantee exclusive access to the
        // selected range for the returned lifetime.
        unsafe { slice::from_raw_parts_mut(ptr, len) }
    }

    fn initialized_len_in_chunk_raw(array: *const Self, chunk: usize) -> usize {
        // SAFETY: caller passes a valid array pointer borrowed from `self`.
        let len = unsafe { (*array).len };
        let start = Self::chunk_start_unchecked(chunk);

        if len <= start {
            return 0;
        }

        let capacity = Self::chunk_capacity_unchecked(chunk);
        (len - start).min(capacity)
    }

    unsafe fn allocated_chunk_pointer_unchecked(
        array: *const Self,
        chunk: usize,
    ) -> *mut MaybeUninit<T> {
        debug_assert!(chunk < CHUNKS);

        // SAFETY: the caller guarantees that `chunk` is within the fixed chunk
        // pointer table and that the selected chunk has already been allocated.
        unsafe {
            (*array)
                .chunks
                .get_unchecked(chunk)
                .unwrap_unchecked()
                .as_ptr()
        }
    }
}

struct DropRangeGuard<T, const BASE_SHIFT: usize, const CHUNKS: usize> {
    array: *mut ExponentialArray<T, BASE_SHIFT, CHUNKS>,
    start: usize,
    end: usize,
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> Drop
    for DropRangeGuard<T, BASE_SHIFT, CHUNKS>
{
    fn drop(&mut self) {
        if self.start == self.end || !mem::needs_drop::<T>() {
            return;
        }

        // SAFETY: guards are created only for initialized ranges whose already
        // consumed prefix is excluded by `start`. During unwinding, `start..end`
        // names only elements that have not been moved out or handed to slice
        // drop glue.
        unsafe { (*self.array).drop_range_unchecked(self.start, self.end) };
    }
}

struct TailInitGuard<T, const BASE_SHIFT: usize, const CHUNKS: usize> {
    array: *mut ExponentialArray<T, BASE_SHIFT, CHUNKS>,
    initialized: usize,
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> TailInitGuard<T, BASE_SHIFT, CHUNKS> {
    fn new(array: &mut ExponentialArray<T, BASE_SHIFT, CHUNKS>) -> Self {
        Self {
            array,
            initialized: 0,
        }
    }

    fn commit(mut self) {
        if self.initialized != 0 {
            // SAFETY: `initialized` counts reserved tail slots this guard has
            // fully initialized and that are not yet included in `len`.
            unsafe { (*self.array).advance_tail_after_bulk_write(self.initialized) };
            self.initialized = 0;
        }
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> Drop
    for TailInitGuard<T, BASE_SHIFT, CHUNKS>
{
    fn drop(&mut self) {
        if self.initialized == 0 {
            return;
        }

        // SAFETY: `initialized` counts reserved tail slots that have been fully
        // initialized before a clone/generator panic. Advancing `len` lets the
        // array's normal drop path destroy them exactly once.
        unsafe { (*self.array).advance_tail_after_bulk_write(self.initialized) };
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> Default
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> Drop
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    fn drop(&mut self) {
        self.clear();

        if mem::size_of::<T>() == 0 {
            return;
        }

        let mut chunk = 0usize;
        while chunk < self.allocated_chunks {
            if let Some(pointer) = self.chunks[chunk] {
                if let Ok(layout) = Self::chunk_layout(chunk) {
                    // SAFETY: chunks are allocated by `try_allocate_next_chunk` with
                    // this exact layout and have not been deallocated yet.
                    unsafe { dealloc(pointer.as_ptr().cast::<u8>(), layout) };
                }
            }
            chunk += 1;
        }
    }
}

impl<T: Clone, const BASE_SHIFT: usize, const CHUNKS: usize> Clone
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    fn clone(&self) -> Self {
        let mut cloned = Self::with_capacity(self.len);
        cloned.extend(self.iter().cloned());
        cloned
    }

    fn clone_from(&mut self, source: &Self) {
        self.clear();
        self.reserve(source.len);
        self.extend(source.iter().cloned());
    }
}

impl<T: fmt::Debug, const BASE_SHIFT: usize, const CHUNKS: usize> fmt::Debug
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

fn next_non_empty_chunk<'a, T, I>(chunks: &mut I) -> Option<&'a [T]>
where
    T: 'a,
    I: Iterator<Item = &'a [T]>,
{
    loop {
        match chunks.next() {
            Some([]) => {}
            chunk => return chunk,
        }
    }
}

fn eq_chunked_sequences<'a, 'b, T, U, I, J>(
    left_len: usize,
    right_len: usize,
    mut left_chunks: I,
    mut right_chunks: J,
) -> bool
where
    T: PartialEq<U> + 'a,
    U: 'b,
    I: Iterator<Item = &'a [T]>,
    J: Iterator<Item = &'b [U]>,
{
    if left_len != right_len {
        return false;
    }

    let mut left = next_non_empty_chunk(&mut left_chunks).unwrap_or(&[]);
    let mut right = next_non_empty_chunk(&mut right_chunks).unwrap_or(&[]);
    while !left.is_empty() || !right.is_empty() {
        let count = left.len().min(right.len());
        if count == 0 {
            return false;
        }
        if left[..count] != right[..count] {
            return false;
        }

        left = &left[count..];
        right = &right[count..];
        if left.is_empty() {
            left = next_non_empty_chunk(&mut left_chunks).unwrap_or(&[]);
        }
        if right.is_empty() {
            right = next_non_empty_chunk(&mut right_chunks).unwrap_or(&[]);
        }
    }

    true
}

fn partial_cmp_chunked_sequences<'a, 'b, T, U, I, J>(
    mut left_chunks: I,
    mut right_chunks: J,
) -> Option<Ordering>
where
    T: PartialOrd<U> + 'a,
    U: 'b,
    I: Iterator<Item = &'a [T]>,
    J: Iterator<Item = &'b [U]>,
{
    let mut left = next_non_empty_chunk(&mut left_chunks).unwrap_or(&[]);
    let mut right = next_non_empty_chunk(&mut right_chunks).unwrap_or(&[]);
    loop {
        match (left.is_empty(), right.is_empty()) {
            (true, true) => return Some(Ordering::Equal),
            (true, false) => return Some(Ordering::Less),
            (false, true) => return Some(Ordering::Greater),
            (false, false) => {}
        }

        let count = left.len().min(right.len());
        let mut offset = 0usize;
        while offset < count {
            match left[offset].partial_cmp(&right[offset])? {
                Ordering::Equal => {}
                ordering => return Some(ordering),
            }
            offset += 1;
        }

        left = &left[count..];
        right = &right[count..];
        if left.is_empty() {
            left = next_non_empty_chunk(&mut left_chunks).unwrap_or(&[]);
        }
        if right.is_empty() {
            right = next_non_empty_chunk(&mut right_chunks).unwrap_or(&[]);
        }
    }
}

fn cmp_chunked_sequences<'a, T, I, J>(mut left_chunks: I, mut right_chunks: J) -> Ordering
where
    T: Ord + 'a,
    I: Iterator<Item = &'a [T]>,
    J: Iterator<Item = &'a [T]>,
{
    let mut left = next_non_empty_chunk(&mut left_chunks).unwrap_or(&[]);
    let mut right = next_non_empty_chunk(&mut right_chunks).unwrap_or(&[]);
    loop {
        match (left.is_empty(), right.is_empty()) {
            (true, true) => return Ordering::Equal,
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            (false, false) => {}
        }

        let count = left.len().min(right.len());
        match left[..count].cmp(&right[..count]) {
            Ordering::Equal => {}
            ordering => return ordering,
        }

        left = &left[count..];
        right = &right[count..];
        if left.is_empty() {
            left = next_non_empty_chunk(&mut left_chunks).unwrap_or(&[]);
        }
        if right.is_empty() {
            right = next_non_empty_chunk(&mut right_chunks).unwrap_or(&[]);
        }
    }
}

impl<
        T,
        U,
        const BASE_SHIFT: usize,
        const CHUNKS: usize,
        const OTHER_BASE_SHIFT: usize,
        const OTHER_CHUNKS: usize,
    > PartialEq<ExponentialArray<U, OTHER_BASE_SHIFT, OTHER_CHUNKS>>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &ExponentialArray<U, OTHER_BASE_SHIFT, OTHER_CHUNKS>) -> bool {
        eq_chunked_sequences(self.len, other.len, self.chunks(), other.chunks())
    }
}

impl<T, U, const BASE_SHIFT: usize, const CHUNKS: usize> PartialEq<[U]>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &[U]) -> bool {
        eq_chunked_sequences(
            self.len,
            other.len(),
            self.chunks(),
            core::iter::once(other),
        )
    }
}

impl<T, U, const BASE_SHIFT: usize, const CHUNKS: usize> PartialEq<&[U]>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &&[U]) -> bool {
        <Self as PartialEq<[U]>>::eq(self, *other)
    }
}

impl<T, U, const N: usize, const BASE_SHIFT: usize, const CHUNKS: usize> PartialEq<[U; N]>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &[U; N]) -> bool {
        <Self as PartialEq<[U]>>::eq(self, &other[..])
    }
}

impl<T, U, const N: usize, const BASE_SHIFT: usize, const CHUNKS: usize> PartialEq<&[U; N]>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &&[U; N]) -> bool {
        <Self as PartialEq<[U]>>::eq(self, &other[..])
    }
}

impl<T, U, const BASE_SHIFT: usize, const CHUNKS: usize> PartialEq<Vec<U>>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &Vec<U>) -> bool {
        <Self as PartialEq<[U]>>::eq(self, other.as_slice())
    }
}

impl<T, U, const BASE_SHIFT: usize, const CHUNKS: usize> PartialEq<&Vec<U>>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &&Vec<U>) -> bool {
        <Self as PartialEq<[U]>>::eq(self, other.as_slice())
    }
}

impl<T: Eq, const BASE_SHIFT: usize, const CHUNKS: usize> Eq
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
}

impl<
        T,
        U,
        const BASE_SHIFT: usize,
        const CHUNKS: usize,
        const OTHER_BASE_SHIFT: usize,
        const OTHER_CHUNKS: usize,
    > PartialOrd<ExponentialArray<U, OTHER_BASE_SHIFT, OTHER_CHUNKS>>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: PartialOrd<U>,
{
    fn partial_cmp(
        &self,
        other: &ExponentialArray<U, OTHER_BASE_SHIFT, OTHER_CHUNKS>,
    ) -> Option<Ordering> {
        partial_cmp_chunked_sequences(self.chunks(), other.chunks())
    }
}

impl<T, U, const BASE_SHIFT: usize, const CHUNKS: usize> PartialOrd<[U]>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: PartialOrd<U>,
{
    fn partial_cmp(&self, other: &[U]) -> Option<Ordering> {
        partial_cmp_chunked_sequences(self.chunks(), core::iter::once(other))
    }
}

impl<T, U, const BASE_SHIFT: usize, const CHUNKS: usize> PartialOrd<&[U]>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: PartialOrd<U>,
{
    fn partial_cmp(&self, other: &&[U]) -> Option<Ordering> {
        <Self as PartialOrd<[U]>>::partial_cmp(self, *other)
    }
}

impl<T, U, const N: usize, const BASE_SHIFT: usize, const CHUNKS: usize> PartialOrd<[U; N]>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: PartialOrd<U>,
{
    fn partial_cmp(&self, other: &[U; N]) -> Option<Ordering> {
        <Self as PartialOrd<[U]>>::partial_cmp(self, &other[..])
    }
}

impl<T, U, const N: usize, const BASE_SHIFT: usize, const CHUNKS: usize> PartialOrd<&[U; N]>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: PartialOrd<U>,
{
    fn partial_cmp(&self, other: &&[U; N]) -> Option<Ordering> {
        <Self as PartialOrd<[U]>>::partial_cmp(self, &other[..])
    }
}

impl<T, U, const BASE_SHIFT: usize, const CHUNKS: usize> PartialOrd<Vec<U>>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: PartialOrd<U>,
{
    fn partial_cmp(&self, other: &Vec<U>) -> Option<Ordering> {
        <Self as PartialOrd<[U]>>::partial_cmp(self, other.as_slice())
    }
}

impl<T, U, const BASE_SHIFT: usize, const CHUNKS: usize> PartialOrd<&Vec<U>>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: PartialOrd<U>,
{
    fn partial_cmp(&self, other: &&Vec<U>) -> Option<Ordering> {
        <Self as PartialOrd<[U]>>::partial_cmp(self, other.as_slice())
    }
}

impl<T: Ord, const BASE_SHIFT: usize, const CHUNKS: usize> Ord
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_chunked_sequences(self.chunks(), other.chunks())
    }
}

impl<T: Hash, const BASE_SHIFT: usize, const CHUNKS: usize> Hash
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len.hash(state);
        for value in self {
            value.hash(state);
        }
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> Extend<T>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();
        if lower != 0 {
            self.reserve(lower);
        }

        for value in iterator {
            self.push(value);
        }
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> Extend<&'a T>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: Copy + 'a,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = &'a T>,
    {
        let mut iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();
        if lower != 0 {
            self.reserve(lower);
            // SAFETY: the reservation above guarantees at least `lower`
            // uninitialized tail slots. The helper handles an iterator that
            // yields fewer items than its lower bound defensively.
            unsafe { self.copy_refs_to_tail_reserved(&mut iterator, lower) };
        }

        for value in iterator {
            self.push(*value);
        }
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> FromIterator<T>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();
        let mut array = Self::with_capacity(lower);
        array.extend(iterator);
        array
    }
}

impl<T, const N: usize, const BASE_SHIFT: usize, const CHUNKS: usize> From<[T; N]>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    fn from(values: [T; N]) -> Self {
        let mut array = Self::with_capacity(N);
        array.extend(values);
        array
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> From<Vec<T>>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    fn from(mut values: Vec<T>) -> Self {
        let len = values.len();
        let mut array = Self::with_capacity(len);
        // SAFETY: `values.as_ptr()..len` is initialized, `array` has enough
        // reserved disjoint tail storage, and `values` is marked empty after
        // the non-panicking bulk move so moved elements are not dropped twice.
        unsafe {
            array.move_from_ptr_to_tail_unchecked(values.as_ptr(), len);
            values.set_len(0);
        }
        array
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> From<ExponentialArray<T, BASE_SHIFT, CHUNKS>>
    for Vec<T>
{
    fn from(mut array: ExponentialArray<T, BASE_SHIFT, CHUNKS>) -> Self {
        let len = array.len();
        let mut values = Vec::with_capacity(len);
        // SAFETY: `array[0..len]` is initialized, `values` has enough reserved
        // disjoint storage, and `array` is marked empty after the non-panicking
        // bulk move so moved elements are not dropped twice.
        unsafe {
            ExponentialArray::<T, BASE_SHIFT, CHUNKS>::move_array_range_to_ptr_unchecked(
                &array,
                0,
                len,
                values.as_mut_ptr(),
            );
            values.set_len(len);
            array.len = 0;
            array.tail_chunk = 0;
            array.tail_offset = 0;
        }
        values
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> From<&[T]>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: Clone,
{
    fn from(values: &[T]) -> Self {
        let mut array = Self::with_capacity(values.len());
        array.extend_from_slice(values);
        array
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> From<&mut [T]>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: Clone,
{
    fn from(values: &mut [T]) -> Self {
        Self::from(&values[..])
    }
}

impl<T, const N: usize, const BASE_SHIFT: usize, const CHUNKS: usize> From<&[T; N]>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: Clone,
{
    fn from(values: &[T; N]) -> Self {
        Self::from(&values[..])
    }
}

impl<T, const N: usize, const BASE_SHIFT: usize, const CHUNKS: usize> From<&mut [T; N]>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
where
    T: Clone,
{
    fn from(values: &mut [T; N]) -> Self {
        Self::from(&values[..])
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> Index<usize>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index) {
            Some(value) => value,
            None => panic!(
                "index {index} out of bounds for ExponentialArray with len {}",
                self.len
            ),
        }
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> IndexMut<usize>
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let len = self.len;
        match self.get_mut(index) {
            Some(value) => value,
            None => panic!("index {index} out of bounds for ExponentialArray with len {len}"),
        }
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> IntoIterator
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    type Item = T;
    type IntoIter = IntoIter<T, BASE_SHIFT, CHUNKS>;

    fn into_iter(mut self) -> Self::IntoIter {
        let len = self.len;
        self.len = 0;
        self.tail_chunk = 0;
        self.tail_offset = 0;

        IntoIter {
            array: self,
            cursor: ElementCursor::new::<T, BASE_SHIFT, CHUNKS>(len),
        }
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> IntoIterator
    for &'a ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T, BASE_SHIFT, CHUNKS>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> IntoIterator
    for &'a mut ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T, BASE_SHIFT, CHUNKS>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[derive(Clone, Copy)]
struct ElementCursor {
    front: usize,
    back: usize,
    front_chunk: usize,
    front_offset: usize,
    front_remaining_in_chunk: usize,
    back_chunk: usize,
    back_offset: usize,
}

impl ElementCursor {
    fn new<T, const BASE_SHIFT: usize, const CHUNKS: usize>(len: usize) -> Self {
        let (back_chunk, back_offset) =
            ExponentialArray::<T, BASE_SHIFT, CHUNKS>::end_position(len);
        Self {
            front: 0,
            back: len,
            front_chunk: 0,
            front_offset: 0,
            front_remaining_in_chunk:
                ExponentialArray::<T, BASE_SHIFT, CHUNKS>::initial_front_remaining(len),
            back_chunk,
            back_offset,
        }
    }

    fn len(&self) -> usize {
        self.back - self.front
    }

    fn next_front<T, const BASE_SHIFT: usize, const CHUNKS: usize>(
        &mut self,
    ) -> Option<(usize, usize)> {
        if self.front == self.back {
            return None;
        }

        let chunk = self.front_chunk;
        let offset = self.front_offset;
        self.front += 1;
        self.front_offset += 1;
        self.front_remaining_in_chunk -= 1;
        if self.front_remaining_in_chunk == 0 && self.front != self.back {
            self.front_chunk += 1;
            self.front_offset = 0;
            self.front_remaining_in_chunk = self.len().min(
                ExponentialArray::<T, BASE_SHIFT, CHUNKS>::chunk_capacity_unchecked(
                    self.front_chunk,
                ),
            );
        }

        Some((chunk, offset))
    }

    fn next_back<T, const BASE_SHIFT: usize, const CHUNKS: usize>(
        &mut self,
    ) -> Option<(usize, usize)> {
        if self.front == self.back {
            return None;
        }

        self.back -= 1;
        if self.back_offset == 0 {
            self.back_chunk -= 1;
            self.back_offset = ExponentialArray::<T, BASE_SHIFT, CHUNKS>::chunk_capacity_unchecked(
                self.back_chunk,
            );
        }
        self.back_offset -= 1;

        Some((self.back_chunk, self.back_offset))
    }
}

/// An iterator over shared references in an [`ExponentialArray`].
pub struct Iter<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> {
    array: *const ExponentialArray<T, BASE_SHIFT, CHUNKS>,
    cursor: ElementCursor,
    marker: PhantomData<&'a T>,
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> Clone
    for Iter<'a, T, BASE_SHIFT, CHUNKS>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> Copy for Iter<'a, T, BASE_SHIFT, CHUNKS> {}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> Iterator
    for Iter<'a, T, BASE_SHIFT, CHUNKS>
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let (chunk, offset) = self.cursor.next_front::<T, BASE_SHIFT, CHUNKS>()?;

        // SAFETY: iterator bounds are initialized indices and each yielded
        // shared reference is valid for `'a`.
        Some(unsafe {
            &*ExponentialArray::ptr_at_chunk_offset_unchecked(self.array, chunk, offset)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut accum = init;
        let mut index = self.cursor.front;
        let end = self.cursor.back;

        while index < end {
            let (chunk, offset) = ExponentialArray::<T, BASE_SHIFT, CHUNKS>::locate(index);
            let chunk_start =
                ExponentialArray::<T, BASE_SHIFT, CHUNKS>::chunk_start_unchecked(chunk);
            let chunk_end = chunk_start
                .saturating_add(
                    ExponentialArray::<T, BASE_SHIFT, CHUNKS>::chunk_capacity_unchecked(chunk),
                )
                .min(end);
            let len = chunk_end - index;

            // SAFETY: `index..chunk_end` is contained in the iterator's
            // initialized remaining range and lies within one chunk.
            let values = unsafe {
                ExponentialArray::chunk_slice_range_from_raw(self.array, chunk, offset, len)
            };
            for value in values {
                accum = f(accum, value);
            }

            index = chunk_end;
        }

        accum
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> DoubleEndedIterator
    for Iter<'a, T, BASE_SHIFT, CHUNKS>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let (chunk, offset) = self.cursor.next_back::<T, BASE_SHIFT, CHUNKS>()?;

        // SAFETY: iterator bounds are initialized indices and each yielded
        // shared reference is valid for `'a`.
        Some(unsafe {
            &*ExponentialArray::ptr_at_chunk_offset_unchecked(self.array, chunk, offset)
        })
    }

    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut accum = init;
        let start = self.cursor.front;
        let mut end = self.cursor.back;

        while start < end {
            let last = end - 1;
            let (chunk, _) = ExponentialArray::<T, BASE_SHIFT, CHUNKS>::locate(last);
            let chunk_start =
                ExponentialArray::<T, BASE_SHIFT, CHUNKS>::chunk_start_unchecked(chunk).max(start);
            let offset = chunk_start
                - ExponentialArray::<T, BASE_SHIFT, CHUNKS>::chunk_start_unchecked(chunk);
            let len = end - chunk_start;

            // SAFETY: `chunk_start..end` is contained in the iterator's
            // initialized remaining range and lies within one chunk.
            let values = unsafe {
                ExponentialArray::chunk_slice_range_from_raw(self.array, chunk, offset, len)
            };
            for value in values.iter().rev() {
                accum = f(accum, value);
            }

            end = chunk_start;
        }

        accum
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> ExactSizeIterator
    for Iter<'a, T, BASE_SHIFT, CHUNKS>
{
    fn len(&self) -> usize {
        self.cursor.len()
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> FusedIterator
    for Iter<'a, T, BASE_SHIFT, CHUNKS>
{
}

/// An iterator over mutable references in an [`ExponentialArray`].
pub struct IterMut<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> {
    array: *mut ExponentialArray<T, BASE_SHIFT, CHUNKS>,
    cursor: ElementCursor,
    marker: PhantomData<&'a mut T>,
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> Iterator
    for IterMut<'a, T, BASE_SHIFT, CHUNKS>
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let (chunk, offset) = self.cursor.next_front::<T, BASE_SHIFT, CHUNKS>()?;

        // SAFETY: the mutable iterator has exclusive access to the array and
        // yields each index at most once.
        Some(unsafe {
            &mut *ExponentialArray::ptr_at_chunk_offset_unchecked(self.array, chunk, offset)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> DoubleEndedIterator
    for IterMut<'a, T, BASE_SHIFT, CHUNKS>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let (chunk, offset) = self.cursor.next_back::<T, BASE_SHIFT, CHUNKS>()?;

        // SAFETY: the mutable iterator has exclusive access to the array and
        // yields each index at most once.
        Some(unsafe {
            &mut *ExponentialArray::ptr_at_chunk_offset_unchecked(self.array, chunk, offset)
        })
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> ExactSizeIterator
    for IterMut<'a, T, BASE_SHIFT, CHUNKS>
{
    fn len(&self) -> usize {
        self.cursor.len()
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> FusedIterator
    for IterMut<'a, T, BASE_SHIFT, CHUNKS>
{
}

/// An iterator over initialized contiguous chunks.
pub struct Chunks<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> {
    array: *const ExponentialArray<T, BASE_SHIFT, CHUNKS>,
    front: usize,
    back: usize,
    marker: PhantomData<&'a [T]>,
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> Clone
    for Chunks<'a, T, BASE_SHIFT, CHUNKS>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> Copy
    for Chunks<'a, T, BASE_SHIFT, CHUNKS>
{
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> Iterator
    for Chunks<'a, T, BASE_SHIFT, CHUNKS>
{
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }

        let chunk = self.front;
        self.front += 1;

        // SAFETY: the chunk range only includes initialized chunks.
        Some(unsafe { ExponentialArray::chunk_slice_from_raw(self.array, chunk) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> DoubleEndedIterator
    for Chunks<'a, T, BASE_SHIFT, CHUNKS>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }

        self.back -= 1;

        // SAFETY: the chunk range only includes initialized chunks.
        Some(unsafe { ExponentialArray::chunk_slice_from_raw(self.array, self.back) })
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> ExactSizeIterator
    for Chunks<'a, T, BASE_SHIFT, CHUNKS>
{
    fn len(&self) -> usize {
        self.back - self.front
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> FusedIterator
    for Chunks<'a, T, BASE_SHIFT, CHUNKS>
{
}

/// An iterator over initialized mutable contiguous chunks.
pub struct ChunksMut<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> {
    array: *mut ExponentialArray<T, BASE_SHIFT, CHUNKS>,
    front: usize,
    back: usize,
    marker: PhantomData<&'a mut [T]>,
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> Iterator
    for ChunksMut<'a, T, BASE_SHIFT, CHUNKS>
{
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }

        let chunk = self.front;
        self.front += 1;

        // SAFETY: the mutable chunk iterator has exclusive access to the array
        // and yields each chunk at most once.
        Some(unsafe { ExponentialArray::chunk_slice_mut_from_raw(self.array, chunk) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> DoubleEndedIterator
    for ChunksMut<'a, T, BASE_SHIFT, CHUNKS>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }

        self.back -= 1;

        // SAFETY: the mutable chunk iterator has exclusive access to the array
        // and yields each chunk at most once.
        Some(unsafe { ExponentialArray::chunk_slice_mut_from_raw(self.array, self.back) })
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> ExactSizeIterator
    for ChunksMut<'a, T, BASE_SHIFT, CHUNKS>
{
    fn len(&self) -> usize {
        self.back - self.front
    }
}

impl<'a, T, const BASE_SHIFT: usize, const CHUNKS: usize> FusedIterator
    for ChunksMut<'a, T, BASE_SHIFT, CHUNKS>
{
}

/// An owning iterator over an [`ExponentialArray`].
pub struct IntoIter<T, const BASE_SHIFT: usize, const CHUNKS: usize> {
    array: ExponentialArray<T, BASE_SHIFT, CHUNKS>,
    cursor: ElementCursor,
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> Iterator for IntoIter<T, BASE_SHIFT, CHUNKS> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let (chunk, offset) = self.cursor.next_front::<T, BASE_SHIFT, CHUNKS>()?;

        // SAFETY: `front..back` contains initialized elements owned by this
        // iterator. Each index is read at most once.
        Some(unsafe {
            ExponentialArray::ptr_at_chunk_offset_unchecked(&self.array, chunk, offset).read()
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> DoubleEndedIterator
    for IntoIter<T, BASE_SHIFT, CHUNKS>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let (chunk, offset) = self.cursor.next_back::<T, BASE_SHIFT, CHUNKS>()?;

        // SAFETY: `front..back` contains initialized elements owned by this
        // iterator. Each index is read at most once.
        Some(unsafe {
            ExponentialArray::ptr_at_chunk_offset_unchecked(&self.array, chunk, offset).read()
        })
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> ExactSizeIterator
    for IntoIter<T, BASE_SHIFT, CHUNKS>
{
    fn len(&self) -> usize {
        self.cursor.len()
    }
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> FusedIterator
    for IntoIter<T, BASE_SHIFT, CHUNKS>
{
}

impl<T, const BASE_SHIFT: usize, const CHUNKS: usize> Drop for IntoIter<T, BASE_SHIFT, CHUNKS> {
    fn drop(&mut self) {
        let front = self.cursor.front;
        let back = self.cursor.back;
        self.cursor.front = back;
        // SAFETY: `front..back` contains initialized elements owned by this
        // iterator and not yet yielded. Advancing `front` first prevents a
        // double drop if element destruction panics.
        unsafe { self.array.drop_range_unchecked(front, back) };
    }
}

#[cold]
#[inline(never)]
fn panic_or_handle_reserve(error: TryReserveError) -> ! {
    match error.kind() {
        TryReserveErrorKind::AllocError { size, align } => {
            let layout = Layout::from_size_align(size, align)
                .expect("allocator error stored an invalid layout");
            handle_alloc_error(layout);
        }
        TryReserveErrorKind::CapacityOverflow | TryReserveErrorKind::CapacityExceeded { .. } => {
            panic!("{error}");
        }
    }
}
