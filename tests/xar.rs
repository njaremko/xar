use std::cell::Cell;
use std::collections::VecDeque;
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

    assert_eq!(xs.into_iter().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6]);
}
