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

## Benchmarks

The crate includes Divan benchmarks comparing `Xar<u64>` against `Vec<u64>` for
append, reserved append, iteration, indexed access, and pop-all workloads:

```sh
cargo bench --bench xar_vs_vec -- --sample-count 30 --skip-ext-time
```

Representative medians from an `aarch64-apple-darwin` release build with
`rustc 1.95.0-nightly (f60a0f1bc 2026-02-02)`, using 1,048,576 elements:

| Workload | `Xar` median | `Vec` median | Shape |
| --- | ---: | ---: | --- |
| `push_empty` | 863.7 us | 1.560 ms | `Xar` avoids contiguous reallocation |
| `push_reserved` | 885.4 us | 1.077 ms | `Xar` appends into allocated chunks |
| `iter_sum` | 289.0 us | 289.2 us | chunk-sliced iteration is Vec-like |
| `indexed_sum` | 1.194 ms | 923.9 us | `Vec` has cheaper indexed addressing |
| `pop_all` | 1.983 ms | 1.191 ms | `Vec` has a simpler contiguous tail |

These numbers are machine-dependent. The intended performance profile is stable
addresses with strong append behavior and efficient chunk iteration, not a
universal replacement for contiguous `Vec` storage.

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

## Acknowledgment

The core segmented exponential-array algorithm is based on Andrew Reece's
discussion of exponential arrays (`xar`):
<https://www.youtube.com/watch?v=i-h95QIGchY>.
