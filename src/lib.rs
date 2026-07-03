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
use core::alloc::Layout;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::{Extend, FromIterator, FusedIterator};
use core::marker::PhantomData;
use core::mem::{self, MaybeUninit};
use core::ops::{Index, IndexMut};
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
/// [`truncate`](Self::truncate), or [`clear`](Self::clear), or when the whole
/// array is dropped. Any raw pointer to that element must then be treated as
/// invalid. For zero-sized types, addresses are not meaningful.
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
            unsafe { ptr::drop_in_place(slice::from_raw_parts_mut(pointer, count)) };
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
        let ptr = if mem::size_of::<T>() == 0 {
            NonNull::<T>::dangling().as_ptr()
        } else {
            // SAFETY: callers only request initialized chunks.
            unsafe { Self::allocated_chunk_pointer_unchecked(array, chunk).cast::<T>() }
        };

        // SAFETY: the chunk contains `len` initialized elements.
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    unsafe fn chunk_slice_mut_from_raw<'a>(array: *mut Self, chunk: usize) -> &'a mut [T] {
        let len = Self::initialized_len_in_chunk_raw(array, chunk);
        let ptr = if mem::size_of::<T>() == 0 {
            NonNull::<T>::dangling().as_ptr()
        } else {
            // SAFETY: callers only request initialized chunks.
            unsafe {
                Self::allocated_chunk_pointer_unchecked(array.cast_const(), chunk).cast::<T>()
            }
        };

        // SAFETY: the chunk contains `len` initialized elements and the mutable
        // chunk iterator yields each chunk at most once.
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

        // SAFETY: the guard is created only by `drop_range_unchecked` while the
        // range names initialized elements. If this runs during unwinding, the
        // current chunk has already been handed to slice drop glue and
        // `start..end` covers only later chunks.
        unsafe { (*self.array).drop_range_unchecked(self.start, self.end) };
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

impl<T: PartialEq, const BASE_SHIFT: usize, const CHUNKS: usize> PartialEq
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

impl<T: Eq, const BASE_SHIFT: usize, const CHUNKS: usize> Eq
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
}

impl<T: PartialOrd, const BASE_SHIFT: usize, const CHUNKS: usize> PartialOrd
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<T: Ord, const BASE_SHIFT: usize, const CHUNKS: usize> Ord
    for ExponentialArray<T, BASE_SHIFT, CHUNKS>
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
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
