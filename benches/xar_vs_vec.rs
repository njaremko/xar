use divan::{black_box, counter::ItemsCount, Bencher};
use xar::Xar;

const SIZES: &[usize] = &[64, 1_024, 16_384];

fn main() {
    divan::main();
}

fn make_xar(len: usize) -> Xar<u64> {
    let mut values = Xar::with_capacity(len);
    for index in 0..len {
        values.push(black_box(index as u64));
    }
    values
}

fn make_vec(len: usize) -> Vec<u64> {
    let mut values = Vec::with_capacity(len);
    for index in 0..len {
        values.push(black_box(index as u64));
    }
    values
}

fn make_indices(len: usize) -> Vec<usize> {
    assert!(len > 0);

    let mut indices = Vec::with_capacity(len);
    let mut index = 0usize;
    for _ in 0..len {
        indices.push(index);
        index = (index + 17) % len;
    }
    indices
}

fn sum_xar(values: &Xar<u64>) -> u64 {
    values
        .iter()
        .fold(0u64, |sum, value| sum.wrapping_add(black_box(*value)))
}

fn sum_vec(values: &[u64]) -> u64 {
    values
        .iter()
        .fold(0u64, |sum, value| sum.wrapping_add(black_box(*value)))
}

fn sum_xar_indices(values: &Xar<u64>, indices: &[usize]) -> u64 {
    let mut sum = 0u64;
    for &index in indices {
        sum = sum.wrapping_add(black_box(values[index]));
    }
    sum
}

fn sum_vec_indices(values: &[u64], indices: &[usize]) -> u64 {
    let mut sum = 0u64;
    for &index in indices {
        sum = sum.wrapping_add(black_box(values[index]));
    }
    sum
}

mod push_empty {
    use super::*;

    #[divan::bench(consts = SIZES)]
    fn xar<const N: usize>(bencher: Bencher) {
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            let mut values = Xar::new();
            for index in 0..N {
                values.push(black_box(index as u64));
            }
            black_box(values)
        });
    }

    #[divan::bench(consts = SIZES)]
    fn vec<const N: usize>(bencher: Bencher) {
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            let mut values = Vec::new();
            for index in 0..N {
                values.push(black_box(index as u64));
            }
            black_box(values)
        });
    }
}

mod push_reserved {
    use super::*;

    #[divan::bench(consts = SIZES)]
    fn xar<const N: usize>(bencher: Bencher) {
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            let mut values = Xar::with_capacity(N);
            for index in 0..N {
                values.push(black_box(index as u64));
            }
            black_box(values)
        });
    }

    #[divan::bench(consts = SIZES)]
    fn vec<const N: usize>(bencher: Bencher) {
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            let mut values = Vec::with_capacity(N);
            for index in 0..N {
                values.push(black_box(index as u64));
            }
            black_box(values)
        });
    }
}

mod iter_sum {
    use super::*;

    #[divan::bench(consts = SIZES)]
    fn xar<const N: usize>(bencher: Bencher) {
        bencher
            .counter(ItemsCount::new(N))
            .with_inputs(|| make_xar(N))
            .bench_local_values(|values| black_box(sum_xar(&values)));
    }

    #[divan::bench(consts = SIZES)]
    fn vec<const N: usize>(bencher: Bencher) {
        bencher
            .counter(ItemsCount::new(N))
            .with_inputs(|| make_vec(N))
            .bench_local_values(|values| black_box(sum_vec(&values)));
    }
}

mod indexed_sum {
    use super::*;

    #[divan::bench(consts = SIZES)]
    fn xar<const N: usize>(bencher: Bencher) {
        bencher
            .counter(ItemsCount::new(N))
            .with_inputs(|| (make_xar(N), make_indices(N)))
            .bench_local_values(|(values, indices)| black_box(sum_xar_indices(&values, &indices)));
    }

    #[divan::bench(consts = SIZES)]
    fn vec<const N: usize>(bencher: Bencher) {
        bencher
            .counter(ItemsCount::new(N))
            .with_inputs(|| (make_vec(N), make_indices(N)))
            .bench_local_values(|(values, indices)| black_box(sum_vec_indices(&values, &indices)));
    }
}

mod pop_all {
    use super::*;

    #[divan::bench(consts = SIZES)]
    fn xar<const N: usize>(bencher: Bencher) {
        bencher
            .counter(ItemsCount::new(N))
            .with_inputs(|| make_xar(N))
            .bench_local_values(|mut values| {
                let mut sum = 0u64;
                while let Some(value) = black_box(&mut values).pop() {
                    sum = sum.wrapping_add(black_box(value));
                }
                black_box(sum)
            });
    }

    #[divan::bench(consts = SIZES)]
    fn vec<const N: usize>(bencher: Bencher) {
        bencher
            .counter(ItemsCount::new(N))
            .with_inputs(|| make_vec(N))
            .bench_local_values(|mut values| {
                let mut sum = 0u64;
                while let Some(value) = black_box(&mut values).pop() {
                    sum = sum.wrapping_add(black_box(value));
                }
                black_box(sum)
            });
    }
}
