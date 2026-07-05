use std::cell::Cell;
use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use hegel::generators as gs;
use xar::{ExponentialArray, TryReserveErrorKind, Xar};

type PropertyArray<T> = ExponentialArray<T, 2, 8>;
type TinyArray<T> = ExponentialArray<T, 0, 3>;

fn draw_values(tc: &hegel::TestCase, max_size: usize) -> Vec<i32> {
    tc.draw(gs::vecs(gs::integers::<i32>()).max_size(max_size))
}

fn expected_chunk_capacity(base_shift: usize, chunk: usize) -> usize {
    let shift = base_shift + chunk.saturating_sub(1);
    assert!(shift < usize::BITS as usize);
    1usize << shift
}

fn expected_allocated_chunks(
    base_shift: usize,
    chunk_count_max: usize,
    requested_capacity: usize,
) -> usize {
    let mut allocated_chunks = 0usize;
    let mut capacity = 0usize;

    while allocated_chunks < chunk_count_max && capacity < requested_capacity {
        capacity += expected_chunk_capacity(base_shift, allocated_chunks);
        allocated_chunks += 1;
    }

    allocated_chunks
}

fn expected_i32_chunks(base_shift: usize, values: &[i32]) -> Vec<Vec<i32>> {
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut chunk = 0usize;

    while start < values.len() {
        let capacity = expected_chunk_capacity(base_shift, chunk);
        let end = values.len().min(start + capacity);
        chunks.push(values[start..end].to_vec());
        start = end;
        chunk += 1;
    }

    chunks
}

fn assert_i32_array_matches_vec<const BASE_SHIFT: usize, const CHUNKS: usize>(
    array: &ExponentialArray<i32, BASE_SHIFT, CHUNKS>,
    expected: &[i32],
) {
    assert_eq!(array.len(), expected.len());
    assert_eq!(array.is_empty(), expected.is_empty());
    assert!(array.capacity() >= array.len());
    assert!(array.allocated_chunks() <= CHUNKS);
    assert_eq!(array.iter().copied().collect::<Vec<_>>(), expected);

    for (index, value) in expected.iter().enumerate() {
        assert_eq!(array.get(index), Some(value));
        assert_eq!(&array[index], value);
        assert!(array.ptr(index).is_some());
    }

    assert_eq!(array.get(expected.len()), None);
    assert_eq!(array.ptr(expected.len()), None);
}

struct XarVecMachine {
    array: PropertyArray<i32>,
    model: Vec<i32>,
}

#[hegel::state_machine]
impl XarVecMachine {
    #[rule]
    fn push(&mut self, tc: hegel::TestCase) {
        let value = tc.draw(gs::integers::<i32>());
        let index = self.array.push(value);
        self.model.push(value);
        assert_eq!(index, self.model.len() - 1);
    }

    #[rule]
    fn try_push(&mut self, tc: hegel::TestCase) {
        let value = tc.draw(gs::integers::<i32>());
        let index = self.array.try_push(value).unwrap();
        self.model.push(value);
        assert_eq!(index, self.model.len() - 1);
    }

    #[rule]
    fn push_with(&mut self, tc: hegel::TestCase) {
        let value = tc.draw(gs::integers::<i32>());
        let calls = Cell::new(0);
        let index = self.array.push_with(|| {
            calls.set(calls.get() + 1);
            value
        });

        self.model.push(value);
        assert_eq!(index, self.model.len() - 1);
        assert_eq!(calls.get(), 1);
    }

    #[rule]
    fn push_mut(&mut self, tc: hegel::TestCase) {
        let initial = tc.draw(gs::integers::<i32>());
        let replacement = tc.draw(gs::integers::<i32>());
        let index = self.model.len();

        {
            let pushed = self.array.push_mut(initial);
            assert_eq!(*pushed, initial);
            *pushed = replacement;
        }

        self.model.push(replacement);
        assert_eq!(self.array.get(index), Some(&replacement));
    }

    #[rule]
    fn push_ptr(&mut self, tc: hegel::TestCase) {
        let initial = tc.draw(gs::integers::<i32>());
        let replacement = tc.draw(gs::integers::<i32>());
        let index = self.model.len();
        let pointer = self.array.push_ptr(initial);

        assert_eq!(unsafe { *pointer.as_ptr() }, initial);
        unsafe { *pointer.as_ptr() = replacement };

        self.model.push(replacement);
        assert_eq!(self.array.ptr(index), Some(pointer));
    }

    #[rule]
    fn pop(&mut self, _: hegel::TestCase) {
        assert_eq!(self.array.pop(), self.model.pop());
    }

    #[rule]
    fn truncate(&mut self, tc: hegel::TestCase) {
        let new_len = tc.draw(gs::integers::<usize>().max_value(self.model.len() + 16));
        self.array.truncate(new_len);
        self.model.truncate(new_len);
    }

    #[rule]
    fn clear(&mut self, _: hegel::TestCase) {
        self.array.clear();
        self.model.clear();
    }

    #[rule]
    fn reserve(&mut self, tc: hegel::TestCase) {
        let additional = tc.draw(gs::integers::<usize>().max_value(64));
        let requested_capacity = self.model.len() + additional;

        self.array.reserve(additional);

        assert_eq!(self.array.len(), self.model.len());
        assert!(self.array.capacity() >= requested_capacity);
    }

    #[rule]
    fn mutate_existing(&mut self, tc: hegel::TestCase) {
        if self.model.is_empty() {
            assert_eq!(self.array.get_mut(0), None);
            return;
        }

        let index = tc.draw(gs::integers::<usize>().max_value(self.model.len() - 1));
        let value = tc.draw(gs::integers::<i32>());

        if tc.draw(gs::booleans()) {
            *self.array.get_mut(index).unwrap() = value;
        } else {
            self.array[index] = value;
        }
        self.model[index] = value;
    }

    #[invariant]
    fn matches_vec(&mut self, _: hegel::TestCase) {
        assert_i32_array_matches_vec(&self.array, &self.model);
    }
}

#[hegel::test(test_cases = 64)]
fn generated_operations_match_vec_model(tc: hegel::TestCase) {
    let machine = XarVecMachine {
        array: PropertyArray::new(),
        model: Vec::new(),
    };

    hegel::stateful::run(machine, tc);
}

#[hegel::test(test_cases = 128)]
fn pushed_values_have_vec_sequence_projection(tc: hegel::TestCase) {
    let values = draw_values(&tc, 300);
    let mut array = PropertyArray::new();

    for (index, value) in values.iter().copied().enumerate() {
        assert_eq!(array.push(value), index);
        assert_eq!(array.len(), index + 1);
    }

    assert_i32_array_matches_vec(&array, &values);
}

#[hegel::test(test_cases = 128)]
fn extend_appends_generated_sequences_in_order(tc: hegel::TestCase) {
    let prefix = draw_values(&tc, 150);
    let suffix = draw_values(&tc, 150);
    let mut expected = prefix.clone();
    expected.extend(suffix.iter().copied());

    let mut array = prefix.into_iter().collect::<PropertyArray<_>>();
    array.extend(suffix);

    assert_i32_array_matches_vec(&array, &expected);
}

#[hegel::test(test_cases = 128)]
fn chunks_partition_the_initialized_sequence(tc: hegel::TestCase) {
    let values = draw_values(&tc, 300);
    let array = values.iter().copied().collect::<PropertyArray<_>>();
    let chunks = array.chunks().collect::<Vec<_>>();

    let flattened = chunks
        .iter()
        .flat_map(|chunk| chunk.iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(flattened, values);

    let total_len = chunks.iter().map(|chunk| chunk.len()).sum::<usize>();
    assert_eq!(total_len, values.len());

    for (chunk_index, chunk) in chunks.iter().enumerate() {
        assert!(!chunk.is_empty());

        let capacity = expected_chunk_capacity(2, chunk_index);
        assert!(chunk.len() <= capacity);

        if chunk_index + 1 < chunks.len() {
            assert_eq!(chunk.len(), capacity);
        }
    }
}

#[hegel::test(test_cases = 128)]
fn chunks_mut_visit_each_element_once(tc: hegel::TestCase) {
    let values = draw_values(&tc, 300);
    let delta = tc.draw(gs::integers::<i32>());
    let mut expected = values.clone();
    let mut array = values.into_iter().collect::<PropertyArray<_>>();

    for chunk in array.chunks_mut() {
        for value in chunk {
            *value = value.wrapping_add(delta);
        }
    }
    for value in &mut expected {
        *value = value.wrapping_add(delta);
    }

    assert_i32_array_matches_vec(&array, &expected);
}

#[hegel::test(test_cases = 128)]
fn chunks_double_ended_iterator_matches_documented_partition(tc: hegel::TestCase) {
    let values = draw_values(&tc, 300);
    let expected_chunks = expected_i32_chunks(2, &values);
    let operations = tc.draw(gs::vecs(gs::booleans()).max_size(expected_chunks.len() + 8));
    let array = values.into_iter().collect::<PropertyArray<_>>();
    let mut expected = VecDeque::from(expected_chunks);
    let mut chunks = array.chunks();

    for take_back in operations {
        assert_eq!(chunks.len(), expected.len());
        let actual = if take_back {
            chunks.next_back().map(<[i32]>::to_vec)
        } else {
            chunks.next().map(<[i32]>::to_vec)
        };
        let expected_chunk = if take_back {
            expected.pop_back()
        } else {
            expected.pop_front()
        };

        assert_eq!(actual, expected_chunk);
    }

    assert_eq!(
        chunks.map(<[i32]>::to_vec).collect::<Vec<_>>(),
        expected.into_iter().collect::<Vec<_>>()
    );
}

#[hegel::test(test_cases = 128)]
fn chunks_mut_double_ended_iterator_visits_each_chunk_once(tc: hegel::TestCase) {
    let values = draw_values(&tc, 300);
    let delta = tc.draw(gs::integers::<i32>());
    let expected_chunks = expected_i32_chunks(2, &values);
    let operations = tc.draw(gs::vecs(gs::booleans()).max_size(expected_chunks.len() + 8));
    let mut expected = values.clone();
    let mut array = values.into_iter().collect::<PropertyArray<_>>();
    let mut expected_queue = VecDeque::from(expected_chunks);
    let mut visited = 0usize;

    {
        let mut chunks = array.chunks_mut();
        for take_back in operations {
            assert_eq!(chunks.len(), expected_queue.len());
            let actual = if take_back {
                chunks.next_back()
            } else {
                chunks.next()
            };
            let expected_chunk = if take_back {
                expected_queue.pop_back()
            } else {
                expected_queue.pop_front()
            };

            match (actual, expected_chunk) {
                (Some(chunk), Some(expected_chunk)) => {
                    assert_eq!(chunk, expected_chunk.as_slice());
                    visited += expected_chunk.len();
                    for value in chunk {
                        *value = value.wrapping_add(delta);
                    }
                }
                (None, None) => {}
                (actual, expected_chunk) => {
                    panic!("chunk iterator mismatch: {actual:?} != {expected_chunk:?}")
                }
            }
        }

        for (chunk, expected_chunk) in chunks.zip(expected_queue) {
            assert_eq!(chunk, expected_chunk.as_slice());
            visited += expected_chunk.len();
            for value in chunk {
                *value = value.wrapping_add(delta);
            }
        }
    }

    for value in &mut expected {
        *value = value.wrapping_add(delta);
    }

    assert_eq!(visited, expected.len());
    assert_i32_array_matches_vec(&array, &expected);
}

#[hegel::test(test_cases = 128)]
fn shared_double_ended_iterator_matches_vec_deque(tc: hegel::TestCase) {
    let values = draw_values(&tc, 300);
    let operations = tc.draw(gs::vecs(gs::booleans()).max_size(values.len() + 16));
    let array = values.iter().copied().collect::<PropertyArray<_>>();
    let mut model = VecDeque::from(values);
    let mut iter = array.iter();

    for take_back in operations {
        assert_eq!(iter.len(), model.len());
        let actual = if take_back {
            iter.next_back().copied()
        } else {
            iter.next().copied()
        };
        let expected = if take_back {
            model.pop_back()
        } else {
            model.pop_front()
        };
        assert_eq!(actual, expected);
    }

    assert_eq!(
        iter.copied().collect::<Vec<_>>(),
        model.into_iter().collect::<Vec<_>>()
    );
}

#[hegel::test(test_cases = 128)]
fn shared_iterator_fold_matches_vec_deque_after_partial_consumption(tc: hegel::TestCase) {
    let values = draw_values(&tc, 300);
    let operations = tc.draw(gs::vecs(gs::booleans()).max_size(values.len() + 16));
    let array = values.iter().copied().collect::<PropertyArray<_>>();

    let mut fold_model = VecDeque::from(values.clone());
    let mut fold_iter = array.iter();
    for &take_back in &operations {
        if take_back {
            assert_eq!(fold_iter.next_back().copied(), fold_model.pop_back());
        } else {
            assert_eq!(fold_iter.next().copied(), fold_model.pop_front());
        }
    }

    let folded = fold_iter.fold(Vec::new(), |mut seen, value| {
        seen.push(*value);
        seen
    });
    assert_eq!(folded, fold_model.iter().copied().collect::<Vec<_>>());

    let mut rfold_model = VecDeque::from(values);
    let mut rfold_iter = array.iter();
    for &take_back in &operations {
        if take_back {
            assert_eq!(rfold_iter.next_back().copied(), rfold_model.pop_back());
        } else {
            assert_eq!(rfold_iter.next().copied(), rfold_model.pop_front());
        }
    }

    let rfolded = rfold_iter.rfold(Vec::new(), |mut seen, value| {
        seen.push(*value);
        seen
    });
    assert_eq!(
        rfolded,
        rfold_model.iter().rev().copied().collect::<Vec<_>>()
    );
}

#[hegel::test(test_cases = 128)]
fn mutable_double_ended_iterator_matches_vec_deque(tc: hegel::TestCase) {
    let values = draw_values(&tc, 300);
    let operations = tc.draw(gs::vecs(gs::booleans()).max_size(values.len() + 16));
    let delta = tc.draw(gs::integers::<i32>());
    let mut expected = values.clone();
    let mut indexes = VecDeque::from_iter(0..values.len());
    let mut array = values.into_iter().collect::<PropertyArray<_>>();

    {
        let mut iter = array.iter_mut();
        for take_back in operations {
            assert_eq!(iter.len(), indexes.len());
            let actual = if take_back {
                iter.next_back()
            } else {
                iter.next()
            };
            let expected_index = if take_back {
                indexes.pop_back()
            } else {
                indexes.pop_front()
            };

            match (actual, expected_index) {
                (Some(value), Some(index)) => {
                    assert_eq!(*value, expected[index]);
                    *value = value.wrapping_add(delta);
                    expected[index] = expected[index].wrapping_add(delta);
                }
                (None, None) => {}
                (actual, expected_index) => {
                    panic!("mutable iterator mismatch: {actual:?} != {expected_index:?}")
                }
            }
        }

        for (value, index) in iter.zip(indexes) {
            assert_eq!(*value, expected[index]);
            *value = value.wrapping_add(delta);
            expected[index] = expected[index].wrapping_add(delta);
        }
    }

    assert_i32_array_matches_vec(&array, &expected);
}

#[hegel::test(test_cases = 128)]
fn owning_double_ended_iterator_drops_each_element_once(tc: hegel::TestCase) {
    struct DropCounter(Rc<Cell<usize>>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    let len = tc.draw(gs::integers::<usize>().max_value(150));
    let operations = tc.draw(gs::vecs(gs::booleans()).max_size(len + 16));
    let drops = Rc::new(Cell::new(0));
    let array = (0..len)
        .map(|_| DropCounter(drops.clone()))
        .collect::<PropertyArray<_>>();
    let mut iter = array.into_iter();

    for take_back in operations {
        if take_back {
            drop(iter.next_back());
        } else {
            drop(iter.next());
        }
        assert!(drops.get() <= len);
    }

    drop(iter);
    assert_eq!(drops.get(), len);
}

#[hegel::test(test_cases = 128)]
fn unchecked_accessors_match_checked_access_for_valid_indices(tc: hegel::TestCase) {
    let values = tc.draw(gs::vecs(gs::integers::<i32>()).min_size(1).max_size(300));
    let index = tc.draw(gs::integers::<usize>().max_value(values.len() - 1));
    let replacement = tc.draw(gs::integers::<i32>());
    let mut expected = values.clone();
    let mut array = values.into_iter().collect::<PropertyArray<_>>();

    assert_eq!(unsafe { *array.get_unchecked(index) }, expected[index]);
    assert_eq!(unsafe { *array.get_unchecked_mut(index) }, expected[index]);

    unsafe { *array.get_unchecked_mut(index) = replacement };
    expected[index] = replacement;

    assert_i32_array_matches_vec(&array, &expected);
}

#[hegel::test(test_cases = 128)]
fn pointers_to_existing_elements_survive_growth_and_reserve(tc: hegel::TestCase) {
    let prefix = tc.draw(gs::vecs(gs::integers::<i64>()).min_size(1).max_size(128));
    let suffix = tc.draw(gs::vecs(gs::integers::<i64>()).max_size(128));
    let additional = tc.draw(gs::integers::<usize>().max_value(256));
    let mut array = PropertyArray::new();

    for value in prefix.iter().copied() {
        array.push(value);
    }

    let pointers = prefix
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (index, array.ptr(index).unwrap(), value))
        .collect::<Vec<_>>();
    let capacity_before = array.capacity();

    array.reserve(additional);
    for value in suffix {
        array.push(value);
    }

    assert!(array.capacity() >= capacity_before);
    for (index, pointer, value) in pointers {
        assert_eq!(array.ptr(index).unwrap(), pointer);
        assert_eq!(unsafe { *pointer.as_ptr() }, value);
    }
}

#[hegel::test(test_cases = 128)]
fn sequence_traits_follow_vec_semantics(tc: hegel::TestCase) {
    let left = draw_values(&tc, 150);
    let right = draw_values(&tc, 150);
    let left_array = left.iter().copied().collect::<PropertyArray<_>>();
    let right_array = right.iter().copied().collect::<PropertyArray<_>>();
    let cloned = left_array.clone();

    assert_i32_array_matches_vec(&cloned, &left);
    assert_eq!(left_array == right_array, left == right);
    assert_eq!(left_array.cmp(&right_array), left.cmp(&right));
    assert_eq!(
        left_array.partial_cmp(&right_array),
        left.partial_cmp(&right)
    );
    assert_eq!(format!("{left_array:?}"), format!("{left:?}"));
}

#[hegel::test(test_cases = 128)]
fn try_reserve_reports_capacity_bounds(tc: hegel::TestCase) {
    let max = TinyArray::<u8>::max_capacity();
    let requested = tc.draw(gs::integers::<usize>().max_value(max + 4));
    let mut array = TinyArray::<u8>::new();
    let result = array.try_reserve(requested);

    assert_eq!(array.len(), 0);
    if requested <= max {
        assert_eq!(result, Ok(()));
        assert!(array.capacity() >= requested);
        assert_eq!(
            array.allocated_chunks(),
            expected_allocated_chunks(0, 3, requested)
        );
    } else {
        assert_eq!(
            result.unwrap_err().kind(),
            TryReserveErrorKind::CapacityExceeded { requested, max }
        );
        assert_eq!(array.capacity(), 0);
        assert_eq!(array.allocated_chunks(), 0);
    }
}

#[hegel::test(test_cases = 128)]
fn try_reserve_overflow_preserves_existing_elements(tc: hegel::TestCase) {
    let max = TinyArray::<u8>::max_capacity();
    let len = tc.draw(gs::integers::<usize>().min_value(1).max_value(max));
    let mut array = TinyArray::new();

    for value in 0..len {
        array.push(value as u8);
    }
    let capacity_before = array.capacity();

    let additional = usize::MAX - len + 1;
    let result = array.try_reserve(additional);

    assert_eq!(
        result.unwrap_err().kind(),
        TryReserveErrorKind::CapacityOverflow
    );
    assert_eq!(array.len(), len);
    assert_eq!(array.capacity(), capacity_before);
    assert_eq!(
        array.iter().copied().collect::<Vec<_>>(),
        (0..len as u8).collect::<Vec<_>>()
    );
}

#[hegel::test(test_cases = 128)]
fn try_with_capacity_reports_capacity_bounds(tc: hegel::TestCase) {
    let max = TinyArray::<u8>::max_capacity();
    let requested = tc.draw(gs::integers::<usize>().max_value(max + 4));
    let result = TinyArray::<u8>::try_with_capacity(requested);

    if requested <= max {
        let array = result.unwrap();
        assert_eq!(array.len(), 0);
        assert!(array.capacity() >= requested);
        assert_eq!(
            array.allocated_chunks(),
            expected_allocated_chunks(0, 3, requested)
        );
    } else {
        assert_eq!(
            result.unwrap_err().kind(),
            TryReserveErrorKind::CapacityExceeded { requested, max }
        );
    }
}

#[hegel::test(test_cases = 128)]
fn try_push_with_calls_closure_only_after_reservation_succeeds(tc: hegel::TestCase) {
    let max = TinyArray::<i32>::max_capacity();
    let len = tc.draw(gs::integers::<usize>().max_value(max));
    let value = tc.draw(gs::integers::<i32>());
    let mut array = TinyArray::new();

    for index in 0..len {
        array.push(index as i32);
    }

    let calls = Cell::new(0);
    let result = array.try_push_with(|| {
        calls.set(calls.get() + 1);
        value
    });

    if len < max {
        assert_eq!(result, Ok(len));
        assert_eq!(calls.get(), 1);
        assert_eq!(array.get(len), Some(&value));
    } else {
        assert_eq!(
            result.unwrap_err().kind(),
            TryReserveErrorKind::CapacityExceeded {
                requested: max + 1,
                max,
            }
        );
        assert_eq!(calls.get(), 0);
        assert_eq!(array.len(), max);
    }
}

#[hegel::test(test_cases = 128)]
fn try_push_returns_original_value_when_capacity_is_full(tc: hegel::TestCase) {
    let max = TinyArray::<i32>::max_capacity();
    let values = tc.draw(gs::vecs(gs::integers::<i32>()).min_size(max).max_size(max));
    let rejected = tc.draw(gs::integers::<i32>());
    let mut array = TinyArray::new();

    for (index, value) in values.iter().copied().enumerate() {
        assert_eq!(array.try_push(value).unwrap(), index);
    }

    let capacity_before = array.capacity();
    let error = array.try_push(rejected).unwrap_err();
    let (value, reserve_error) = error.into_parts();

    assert_eq!(value, rejected);
    assert_eq!(
        reserve_error.kind(),
        TryReserveErrorKind::CapacityExceeded {
            requested: max + 1,
            max,
        }
    );
    assert_eq!(array.len(), max);
    assert_eq!(array.capacity(), capacity_before);
    assert_eq!(array.iter().copied().collect::<Vec<_>>(), values);
}

#[test]
fn push_and_get_by_index() {
    let mut xs = Xar::new();

    for i in 0..1_000 {
        assert_eq!(xs.push(i), i);
    }

    assert_eq!(xs.len(), 1_000);
    assert!(!xs.is_empty());

    for i in 0..1_000 {
        assert_eq!(xs.get(i), Some(&i));
        assert_eq!(xs[i], i);
    }

    assert_eq!(xs.get(1_000), None);
}

#[test]
fn pointers_remain_stable_across_growth() {
    let mut xs = Xar::new();

    let first = xs.push_ptr(String::from("first"));
    let first_addr = first.as_ptr() as usize;

    for i in 0..20_000 {
        xs.push(i.to_string());
    }

    assert_eq!(first.as_ptr() as usize, first_addr);
    assert_eq!(unsafe { first.as_ref() }, "first");
}

#[test]
fn reserve_does_not_move_existing_elements() {
    let mut xs = Xar::new();
    xs.push(10);
    let ptr_before = xs.ptr(0).unwrap();

    xs.reserve(10_000);

    let ptr_after = xs.ptr(0).unwrap();
    assert_eq!(ptr_before, ptr_after);
    assert_eq!(xs[0], 10);
}

#[test]
fn pop_and_truncate_drop_tail() {
    #[derive(Clone)]
    struct DropCounter(Rc<Cell<usize>>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    let drops = Rc::new(Cell::new(0));
    let mut xs = Xar::new();

    for _ in 0..10 {
        xs.push(DropCounter(drops.clone()));
    }

    assert_eq!(drops.get(), 0);
    drop(xs.pop());
    assert_eq!(drops.get(), 1);

    xs.truncate(4);
    assert_eq!(xs.len(), 4);
    assert_eq!(drops.get(), 6);

    drop(xs);
    assert_eq!(drops.get(), 10);
}

#[test]
fn push_resumes_at_tail_after_pop_truncate_and_clear() {
    let mut xs = (0..10).collect::<ExponentialArray<_, 2, 4>>();

    assert_eq!(xs.pop(), Some(9));
    assert_eq!(xs.pop(), Some(8));
    assert_eq!(xs.push(90), 8);
    assert_eq!(xs.push(91), 9);
    assert_eq!(
        xs.iter().copied().collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 90, 91]
    );

    xs.truncate(5);
    assert_eq!(xs.push(50), 5);
    assert_eq!(xs.push(51), 6);
    assert_eq!(xs.push(52), 7);
    assert_eq!(xs.push(80), 8);
    assert_eq!(
        xs.chunks().map(|chunk| chunk.to_vec()).collect::<Vec<_>>(),
        vec![vec![0, 1, 2, 3], vec![4, 50, 51, 52], vec![80]]
    );

    xs.clear();
    assert_eq!(xs.push(7), 0);
    assert_eq!(xs.iter().copied().collect::<Vec<_>>(), vec![7]);
}

#[test]
fn append_constructors_append_and_observe_tail_element() {
    let mut xs = ExponentialArray::<_, 2, 4>::new();

    assert_eq!(xs.push(10), 0);

    let pushed_mut = xs.push_mut(20);
    assert_eq!(*pushed_mut, 20);
    *pushed_mut = 21;

    let pushed_ptr = xs.push_ptr(30);
    assert_eq!(unsafe { *pushed_ptr.as_ptr() }, 30);
    unsafe { *pushed_ptr.as_ptr() = 31 };

    assert_eq!(xs.push_with(|| 40), 3);
    assert_eq!(xs.try_push(50).unwrap(), 4);
    assert_eq!(xs.try_push_with(|| 60).unwrap(), 5);
    assert_eq!(
        xs.iter().copied().collect::<Vec<_>>(),
        vec![10, 21, 31, 40, 50, 60]
    );
}

#[test]
fn mutable_double_ended_iterator_matches_vec_deque_order() {
    let mut xs = (0..10).collect::<ExponentialArray<_, 2, 4>>();
    let mut model = VecDeque::from_iter(0..10);
    let operations = [false, true, false, true, true, false, false, true];

    let mut observed = Vec::new();
    let mut expected_observed = Vec::new();
    {
        let mut iter = xs.iter_mut();
        for take_back in operations {
            assert_eq!(iter.len(), model.len());
            let expected = if take_back {
                model.pop_back().unwrap()
            } else {
                model.pop_front().unwrap()
            };
            let value = if take_back {
                let value = iter.next_back().unwrap();
                assert_eq!(*value, expected);
                value
            } else {
                let value = iter.next().unwrap();
                assert_eq!(*value, expected);
                value
            };
            expected_observed.push(expected);
            observed.push(*value);
            *value += 100;
        }

        let remaining = iter
            .map(|value| {
                let old_value = *value;
                *value += 100;
                old_value
            })
            .collect::<Vec<_>>();
        expected_observed.extend(model);
        observed.extend(remaining);
    }

    assert_eq!(observed, expected_observed);
    assert_eq!(
        xs.iter().copied().collect::<Vec<_>>(),
        (100..110).collect::<Vec<_>>()
    );
}

#[test]
fn owning_double_ended_iterator_matches_vec_deque_order() {
    let xs = (0..10).collect::<ExponentialArray<_, 2, 4>>();
    let mut model = VecDeque::from_iter(0..10);
    let operations = [false, true, true, false, false, true];
    let mut iter = xs.into_iter();

    for take_back in operations {
        assert_eq!(iter.len(), model.len());
        let actual = if take_back {
            iter.next_back()
        } else {
            iter.next()
        };
        let expected = if take_back {
            model.pop_back()
        } else {
            model.pop_front()
        };
        assert_eq!(actual, expected);
    }

    assert_eq!(
        iter.collect::<Vec<_>>(),
        model.into_iter().collect::<Vec<_>>()
    );
}

#[test]
fn clear_drops_later_chunks_when_destructor_panics() {
    struct PanicOnce {
        id: usize,
        panicked: Rc<Cell<bool>>,
        drops: Rc<std::cell::RefCell<Vec<usize>>>,
    }

    impl Drop for PanicOnce {
        fn drop(&mut self) {
            self.drops.borrow_mut().push(self.id);
            if self.id == 0 && !self.panicked.replace(true) {
                panic!("drop panic for test");
            }
        }
    }

    let panicked = Rc::new(Cell::new(false));
    let drops = Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut xs = ExponentialArray::<_, 1, 4>::new();
    for id in 0..6 {
        xs.push(PanicOnce {
            id,
            panicked: panicked.clone(),
            drops: drops.clone(),
        });
    }

    let result = catch_unwind(AssertUnwindSafe(|| xs.clear()));
    assert!(result.is_err());

    let mut actual = drops.borrow().clone();
    actual.sort_unstable();
    assert_eq!(actual, vec![0, 1, 2, 3, 4, 5]);

    drop(xs);
    let mut actual = drops.borrow().clone();
    actual.sort_unstable();
    assert_eq!(actual, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn iterators_work_from_both_ends() {
    let xs = (0..10).collect::<Xar<_>>();
    let mut iter = xs.iter();

    assert_eq!(iter.next(), Some(&0));
    assert_eq!(iter.next_back(), Some(&9));
    assert_eq!(iter.len(), 8);
    assert_eq!(
        iter.copied().collect::<Vec<_>>(),
        (1..9).collect::<Vec<_>>()
    );
}

#[test]
fn mutable_iterators_work_from_both_ends() {
    let mut xs = (0..10).collect::<Xar<_>>();

    for value in xs.iter_mut() {
        *value *= 2;
    }

    assert_eq!(
        xs.into_iter().collect::<Vec<_>>(),
        (0..10).map(|v| v * 2).collect::<Vec<_>>()
    );
}

#[test]
fn chunks_report_contiguous_initialized_segments() {
    let xs = (0..10).collect::<ExponentialArray<_, 2, 4>>();
    let chunks = xs.chunks().map(|chunk| chunk.to_vec()).collect::<Vec<_>>();

    assert_eq!(chunks, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9]]);
}

#[test]
fn chunks_follow_power_range_layout() {
    let xs = (0..32).collect::<ExponentialArray<_, 2, 5>>();
    let chunks = xs.chunks().map(|chunk| chunk.to_vec()).collect::<Vec<_>>();

    assert_eq!(
        chunks,
        vec![
            (0..4).collect::<Vec<_>>(),
            (4..8).collect::<Vec<_>>(),
            (8..16).collect::<Vec<_>>(),
            (16..32).collect::<Vec<_>>(),
        ]
    );
}

#[test]
fn chunks_mut_can_modify_in_place() {
    let mut xs = (0..10).collect::<ExponentialArray<_, 2, 4>>();

    for chunk in xs.chunks_mut() {
        for value in chunk {
            *value += 1;
        }
    }

    assert_eq!(
        xs.into_iter().collect::<Vec<_>>(),
        (1..11).collect::<Vec<_>>()
    );
}

#[test]
fn clone_debug_eq_ord_and_hash_are_sequence_like() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let xs = (0..32).collect::<Xar<_>>();
    let ys = xs.clone();

    assert_eq!(xs, ys);
    assert_eq!(
        format!("{xs:?}"),
        format!("{:?}", (0..32).collect::<Vec<_>>())
    );
    assert!(xs <= ys);

    let mut hx = DefaultHasher::new();
    let mut hy = DefaultHasher::new();
    xs.hash(&mut hx);
    ys.hash(&mut hy);
    assert_eq!(hx.finish(), hy.finish());
}

#[test]
fn supports_zero_sized_types() {
    let mut xs = Xar::new();

    for _ in 0..128 {
        xs.push(());
    }

    assert_eq!(xs.len(), 128);
    assert_eq!(xs.iter().count(), 128);
    assert_eq!(xs.pop(), Some(()));
    assert_eq!(xs.len(), 127);
}

#[test]
fn front_and_back_helpers_match_vec_observations() {
    let mut xs = PropertyArray::<i32>::new();

    assert_eq!(xs.first(), None);
    assert_eq!(xs.first_mut(), None);
    assert_eq!(xs.last(), None);
    assert_eq!(xs.last_mut(), None);
    assert_eq!(xs.last_ptr(), None);

    xs.extend([1, 2, 3, 4]);
    assert_eq!(xs.first(), Some(&1));
    assert_eq!(xs.last(), Some(&4));

    *xs.first_mut().unwrap() = 10;
    *xs.last_mut().unwrap() = 40;

    let last_pointer = xs.last_ptr().unwrap();
    assert_eq!(last_pointer, xs.ptr(xs.len() - 1).unwrap());
    assert_eq!(unsafe { *last_pointer.as_ptr() }, 40);
    assert_i32_array_matches_vec(&xs, &[10, 2, 3, 40]);
}

#[test]
fn append_moves_source_sequence_without_moving_existing_destination() {
    let mut dst = (0..8).collect::<PropertyArray<_>>();
    let stable_prefix = (0..dst.len())
        .map(|index| (index, dst.ptr(index).unwrap(), dst[index]))
        .collect::<Vec<_>>();
    let mut src = (100..108).collect::<PropertyArray<_>>();
    let expected = (0..8).chain(100..108).collect::<Vec<_>>();

    dst.append(&mut src);

    assert!(src.is_empty());
    assert_i32_array_matches_vec(&dst, &expected);
    for (index, pointer, value) in stable_prefix {
        assert_eq!(dst.ptr(index).unwrap(), pointer);
        assert_eq!(unsafe { *pointer.as_ptr() }, value);
    }
}

#[test]
fn append_capacity_failure_leaves_both_sequences_unchanged() {
    let mut dst = ExponentialArray::<u8, 0, 2>::from([1, 2]);
    let mut src = ExponentialArray::<u8, 0, 2>::from([3]);

    let result = catch_unwind(AssertUnwindSafe(|| dst.append(&mut src)));

    assert!(result.is_err());
    assert_eq!(dst.iter().copied().collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(src.iter().copied().collect::<Vec<_>>(), vec![3]);
}

#[test]
fn resize_and_resize_with_match_vec_tail_behavior() {
    let mut xs = PropertyArray::from([1, 2, 3, 4]);
    let first_pointer = xs.ptr(0).unwrap();

    xs.resize(2, 9);
    assert_i32_array_matches_vec(&xs, &[1, 2]);
    assert_eq!(xs.ptr(0).unwrap(), first_pointer);

    xs.resize(5, 7);
    assert_i32_array_matches_vec(&xs, &[1, 2, 7, 7, 7]);
    assert_eq!(xs.ptr(0).unwrap(), first_pointer);

    let mut next_value = 10;
    xs.resize_with(8, || {
        let value = next_value;
        next_value += 1;
        value
    });
    assert_i32_array_matches_vec(&xs, &[1, 2, 7, 7, 7, 10, 11, 12]);
    assert_eq!(xs.ptr(0).unwrap(), first_pointer);

    xs.resize_with(3, || panic!("truncate must not call the generator"));
    assert_i32_array_matches_vec(&xs, &[1, 2, 7]);
    assert_eq!(xs.ptr(0).unwrap(), first_pointer);
}

#[test]
fn resize_capacity_failure_preserves_existing_sequence() {
    let mut resized = ExponentialArray::<i32, 0, 2>::from([1, 2]);
    let resize_result = catch_unwind(AssertUnwindSafe(|| resized.resize(3, 9)));
    assert!(resize_result.is_err());
    assert_eq!(resized.iter().copied().collect::<Vec<_>>(), vec![1, 2]);

    let mut resized_with = ExponentialArray::<i32, 0, 2>::from([1, 2]);
    let calls = Cell::new(0);
    let resize_with_result = catch_unwind(AssertUnwindSafe(|| {
        resized_with.resize_with(3, || {
            calls.set(calls.get() + 1);
            9
        })
    }));

    assert!(resize_with_result.is_err());
    assert_eq!(calls.get(), 0);
    assert_eq!(resized_with.iter().copied().collect::<Vec<_>>(), vec![1, 2]);
}

#[test]
fn split_off_moves_tail_without_moving_retained_prefix() {
    let mut xs = (0..10).collect::<PropertyArray<_>>();
    let stable_prefix = (0..4)
        .map(|index| (index, xs.ptr(index).unwrap(), xs[index]))
        .collect::<Vec<_>>();

    let tail = xs.split_off(4);

    assert_i32_array_matches_vec(&xs, &[0, 1, 2, 3]);
    assert_i32_array_matches_vec(&tail, &[4, 5, 6, 7, 8, 9]);
    for (index, pointer, value) in stable_prefix {
        assert_eq!(xs.ptr(index).unwrap(), pointer);
        assert_eq!(unsafe { *pointer.as_ptr() }, value);
    }

    let empty_tail = xs.split_off(xs.len());
    assert!(empty_tail.is_empty());
    assert_i32_array_matches_vec(&xs, &[0, 1, 2, 3]);

    let all = xs.split_off(0);
    assert!(xs.is_empty());
    assert_i32_array_matches_vec(&all, &[0, 1, 2, 3]);
}

#[test]
fn split_off_out_of_bounds_preserves_sequence() {
    let mut xs = PropertyArray::from([1, 2, 3]);
    let result = catch_unwind(AssertUnwindSafe(|| xs.split_off(4)));

    assert!(result.is_err());
    assert_i32_array_matches_vec(&xs, &[1, 2, 3]);
}

#[test]
fn extend_from_within_matches_vec_and_preserves_existing_addresses() {
    let mut xs = PropertyArray::from([0, 1, 2, 3, 4]);
    let stable_prefix = (0..xs.len())
        .map(|index| (index, xs.ptr(index).unwrap(), xs[index]))
        .collect::<Vec<_>>();
    let mut model = vec![0, 1, 2, 3, 4];

    xs.extend_from_within(1..4);
    model.extend_from_within(1..4);
    assert_i32_array_matches_vec(&xs, &model);

    xs.extend_from_within(..=2);
    model.extend_from_within(..=2);
    assert_i32_array_matches_vec(&xs, &model);

    for (index, pointer, value) in stable_prefix {
        assert_eq!(xs.ptr(index).unwrap(), pointer);
        assert_eq!(unsafe { *pointer.as_ptr() }, value);
    }
}

#[test]
fn extend_from_within_invalid_range_preserves_sequence() {
    let mut xs = PropertyArray::from([1, 2, 3]);
    let reversed_start = 2;
    let reversed_end = 1;
    let reversed = catch_unwind(AssertUnwindSafe(|| {
        xs.extend_from_within(reversed_start..reversed_end)
    }));
    assert!(reversed.is_err());
    assert_i32_array_matches_vec(&xs, &[1, 2, 3]);

    let out_of_bounds = catch_unwind(AssertUnwindSafe(|| xs.extend_from_within(1..4)));
    assert!(out_of_bounds.is_err());
    assert_i32_array_matches_vec(&xs, &[1, 2, 3]);
}

#[test]
fn slice_array_and_vec_conversions_preserve_sequence() {
    let source = vec![String::from("a"), String::from("b")];
    let from_slice = PropertyArray::<String>::from(source.as_slice());
    assert_eq!(from_slice.iter().cloned().collect::<Vec<_>>(), source);

    let mut mutable_slice_values = [String::from("c"), String::from("d")];
    let from_mut_slice = PropertyArray::<String>::from(&mut mutable_slice_values[..]);
    assert_eq!(
        from_mut_slice.iter().cloned().collect::<Vec<_>>(),
        vec![String::from("c"), String::from("d")]
    );

    let from_array_ref = PropertyArray::<String>::from(&[String::from("e"), String::from("f")]);
    assert_eq!(
        from_array_ref.iter().cloned().collect::<Vec<_>>(),
        vec![String::from("e"), String::from("f")]
    );

    let mut mutable_array_values = [String::from("g"), String::from("h")];
    let from_mut_array = PropertyArray::<String>::from(&mut mutable_array_values);
    assert_eq!(
        from_mut_array.iter().cloned().collect::<Vec<_>>(),
        vec![String::from("g"), String::from("h")]
    );

    let from_vec = PropertyArray::<String>::from(vec![String::from("i"), String::from("j")]);
    let back_to_vec: Vec<_> = from_vec.into();
    assert_eq!(back_to_vec, vec![String::from("i"), String::from("j")]);
}

#[test]
fn comparison_and_extend_traits_accept_common_sequence_inputs() {
    let mut xs = PropertyArray::<i32>::new();
    xs.extend(&[1, 2, 3]);

    let same_array = [1, 2, 3];
    let same_slice: &[i32] = &same_array;
    let same_vec = vec![1, 2, 3];
    let larger_vec = vec![1, 2, 4];
    let same_other_config = ExponentialArray::<_, 1, 8>::from([1, 2, 3]);

    assert_eq!(xs, same_array);
    assert_eq!(xs, same_slice);
    assert_eq!(xs, same_vec);
    assert_eq!(xs, same_other_config);
    assert!(xs <= same_slice);
    assert!(xs < larger_vec);
}

#[test]
fn capacity_errors_return_original_value() {
    let mut xs = ExponentialArray::<u8, 0, 2>::new();
    assert_eq!(ExponentialArray::<u8, 0, 2>::max_capacity(), 2);

    assert_eq!(xs.push(1), 0);
    assert_eq!(xs.push(2), 1);

    let error = xs.try_push(3).unwrap_err();
    let (value, reserve_error) = error.into_parts();

    assert_eq!(value, 3);
    assert_eq!(
        reserve_error.kind(),
        TryReserveErrorKind::CapacityExceeded {
            requested: 3,
            max: 2,
        }
    );
}

#[test]
fn from_array_and_extend() {
    let mut xs = ExponentialArray::<_, 1, 8>::from([1, 2, 3]);
    xs.extend([4, 5, 6]);
    xs.extend(&[7, 8, 9]);

    assert_eq!(
        xs.into_iter().collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
}
