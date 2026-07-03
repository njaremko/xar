# xar

`xar` is a Rust implementation of an exponential segmented array: a `Vec`-like
appendable collection whose elements live in independently allocated chunks. The
first two chunks each have `1 << BASE_SHIFT` slots, then each later chunk
doubles. Existing chunks are never reallocated, so element addresses remain
stable while those elements remain initialized.

This is useful for append-heavy collections such as AST nodes, IR nodes, graph
nodes, editor objects, or other object stores where external structures keep raw
pointers to elements.

It is not a contiguous array. Use `chunks()` / `chunks_mut()` when you need
contiguous slices for interop.

## Example

```rust
use xar::Xar;

let mut nodes = Xar::new();
let root = nodes.push_ptr(String::from("root"));

for i in 0..10_000 {
    nodes.push(i.to_string());
}

assert_eq!(unsafe { root.as_ref() }, "root");
```

Dereferencing a pointer returned by `ptr()` or `push_ptr()` is still `unsafe`.
The crate guarantees address stability, not aliasing correctness. A pointer must
not be used after the pointed-to element is removed by `pop`, `truncate`,
`clear`, or after the array is dropped.

## API shape

The public API follows common Rust container conventions:

- `new`, `with_capacity`, `reserve`, and `try_reserve`
- `push`, `try_push`, `pop`, `truncate`, and `clear`
- `get`, `get_mut`, `Index`, and `IndexMut`
- `iter`, `iter_mut`, `chunks`, `chunks_mut`
- `IntoIterator` for owned, shared, and mutable forms
- common traits: `Default`, `Clone`, `Debug`, `Eq`, `Ord`, `Hash`, `Extend`, and `FromIterator`

`push` returns the appended element's index. Use `push_mut` for an immediate
mutable reference or `push_ptr` for an immediate stable raw pointer.

## Configuration

`Xar<T>` is a type alias:

```rust
pub type Xar<T> = ExponentialArray<T, DEFAULT_BASE_SHIFT, DEFAULT_CHUNKS>;
```

For a smaller first chunk or a lower maximum capacity, use `ExponentialArray`
directly:

```rust
use xar::ExponentialArray;

// First two chunks: 4 elements each. Total chunks: 8.
let mut xs = ExponentialArray::<u32, 2, 8>::new();
```

## `no_std`

The crate supports `no_std` with `alloc`:

```toml
xar-array = { version = "0.1", default-features = false }
```

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT license
