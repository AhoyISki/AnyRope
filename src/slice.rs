use std::{cmp::Ordering, ops::RangeBounds, sync::Arc};

use crate::{
    end_bound_to_num, fallible_saturating_sub,
    iter::{Chunks, Iter},
    measures_from_range,
    rope::Rope,
    slice_utils::{end_measure_to_index, index_to_measure, measure_of, start_measure_to_index},
    start_bound_to_num,
    tree::{max_children, max_len, Count, Node, SliceInfo},
    Error, FallibleOrd, Measurable, MeasureRange, Result,
};

/// An immutable view into part of a [`Rope<M>`].
///
/// Just like standard [`&[M]`][Measurable] slices, [`RopeSlice<M>`]s behave as
/// if the slice in their range is the only slice that exists. All indexing is
/// relative to the start of their range, and all iterators and methods that
/// return [`M`][Measurable]s truncate those to the range of the slice.
///
/// In other words, the behavior of a [`RopeSlice<M>`] is always identical to
/// that of a full [`Rope<M>`] created from the same slice range. Nothing should
/// be surprising here.
#[derive(Copy, Clone)]
pub struct RopeSlice<'a, M>(pub(crate) RSEnum<'a, M>)
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized;

#[derive(Copy, Clone, Debug)]
pub(crate) enum RSEnum<'a, M>
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    Full {
        node: &'a Arc<Node<M>>,
        start_info: SliceInfo<M::Measure>,
        end_info: SliceInfo<M::Measure>,
    },
    Light {
        slice: &'a [M],
    },
}

impl<'a, M> RopeSlice<'a, M>
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    pub(crate) fn new_with_range(
        node: &'a Arc<Node<M>>,
        start: M::Measure,
        end: M::Measure,
        cmp: &impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Self {
        // Early-out shortcut for taking a slice of the full thing.
        if cmp(&start, &M::Measure::default()).is_eq() && cmp(&end, &node.measure()).is_eq() {
            if node.is_leaf() {
                let slice = node.leaf_slice();
                return RopeSlice(RSEnum::Light { slice });
            } else {
                return RopeSlice(RSEnum::Full {
                    node,
                    start_info: SliceInfo {
                        len: 0,
                        measure: M::Measure::default(),
                    },
                    end_info: SliceInfo {
                        len: node.len() as Count,
                        measure: node.measure(),
                    },
                });
            }
        }

        // Find the deepest node that still contains the full range given.
        let mut n_start = start;
        let mut n_end = end;
        let mut node = node;
        'outer: loop {
            match *(node as &Node<M>) {
                // Early out if we reach a leaf, because we can do the
                // simpler lightweight slice then.
                Node::Leaf(ref slice) => {
                    let start = start_measure_to_index(slice, n_start, cmp);
                    let end = start + end_measure_to_index(&slice[start..], n_end - n_start, cmp);
                    return RopeSlice(RSEnum::Light {
                        slice: &slice[start..end],
                    });
                }

                Node::Branch(ref children) => {
                    let mut start_measure = M::Measure::default();
                    for (i, info) in children.info().iter().enumerate() {
                        if cmp(&n_start, &start_measure).is_gt()
                            && cmp(&n_end, &(start_measure + info.measure)).is_lt()
                        {
                            n_start = n_start - start_measure;
                            n_end = n_end - start_measure;
                            node = &children.nodes()[i];
                            continue 'outer;
                        }
                        start_measure = start_measure + info.measure;
                    }
                    break;
                }
            }
        }

        // Create the slice
        RopeSlice(RSEnum::Full {
            node,
            start_info: node.start_measure_to_slice_info(n_start, cmp),
            end_info: node.end_measure_to_slice_info(n_end, cmp),
        })
    }

    pub(crate) fn new_with_index_range(
        node: &'a Arc<Node<M>>,
        start: usize,
        end: usize,
    ) -> Result<Self, M> {
        assert!(start <= end);
        assert!(end <= node.info().len as usize);

        // Early-out shortcut for taking a slice of the full thing.
        if start == 0 && end == node.len() {
            if node.is_leaf() {
                let slice = node.leaf_slice();
                return Ok(RopeSlice(RSEnum::Light { slice }));
            } else {
                return Ok(RopeSlice(RSEnum::Full {
                    node,
                    start_info: SliceInfo {
                        len: 0,
                        measure: M::Measure::default(),
                    },
                    end_info: SliceInfo {
                        len: node.len() as Count,
                        measure: node.measure(),
                    },
                }));
            }
        }

        // Find the deepest node that still contains the full range given.
        let mut n_start = start;
        let mut n_end = end;
        let mut node = node;
        'outer: loop {
            match *(node as &Node<M>) {
                // Early out if we reach a leaf, because we can do the
                // simpler lightweight slice then.
                Node::Leaf(ref slice) => {
                    let start_index = n_start;
                    let end_index = n_end;
                    return Ok(RopeSlice(RSEnum::Light {
                        slice: &slice[start_index..end_index],
                    }));
                }

                Node::Branch(ref children) => {
                    let mut start_index = 0;
                    for (i, info) in children.info().iter().enumerate() {
                        if n_start >= start_index && n_end <= (start_index + info.len as usize) {
                            n_start -= start_index;
                            n_end -= start_index;
                            node = &children.nodes()[i];
                            continue 'outer;
                        }
                        start_index += info.len as usize;
                    }
                    break;
                }
            }
        }

        // Create the slice
        Ok(RopeSlice(RSEnum::Full {
            node,
            start_info: node.index_to_slice_info(n_start),
            end_info: node.index_to_slice_info(n_end),
        }))
    }

    //-----------------------------------------------------------------------
    // Informational methods

    /// Total number of elements in [`RopeSlice<M>`].
    ///
    /// Runs in O(1) time.
    #[inline]
    pub fn len(&self) -> usize {
        match *self {
            RopeSlice(RSEnum::Full {
                end_info,
                start_info,
                ..
            }) => (end_info.len - start_info.len) as usize,
            RopeSlice(RSEnum::Light { slice }) => slice.len(),
        }
    }

    /// Returns `true` if the [`RopeSlice<M>`] is empty.
    ///
    /// Runs in O(1) time.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sum of all measures of in [`RopeSlice<M>`].
    ///
    /// Runs in O(1) time.
    #[inline]
    pub fn measure(&self) -> M::Measure {
        match *self {
            RopeSlice(RSEnum::Full {
                end_info,
                start_info,
                ..
            }) => end_info.measure - start_info.measure,
            RopeSlice(RSEnum::Light { slice }) => measure_of(slice),
        }
    }

    //-----------------------------------------------------------------------
    // Index conversion methods

    /// Returns the measure sum at the start of the given index.
    ///
    /// Notes:
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `index` is out of bounds (i.e. `index > Rope::len()`).
    #[inline]
    pub fn index_to_measure(&self, index: usize) -> M::Measure {
        self.try_index_to_measure(index).unwrap()
    }

    /// Returns an index, given a starting measure sum.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `measure` is out of bounds (i.e. `measure >
    /// RopeSlice::measure()`).
    #[inline]
    pub fn start_measure_to_index(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> usize {
        self.try_start_measure_to_index(measure, cmp).unwrap()
    }

    /// Returns an index, given an ending measure sum.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `measure` is out of bounds (i.e. `measure >
    /// RopeSlice::measure()`).
    #[inline]
    pub fn end_measure_to_index(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> usize {
        self.try_end_measure_to_index(measure, cmp).unwrap()
    }

    //-----------------------------------------------------------------------
    // Fetch methods

    /// Returns the [`M`][Measurable] at `index` and the starting measure sum of
    /// that element.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `index` is out of bounds (i.e. `index >
    /// RopeSlice::len()`).
    #[inline]
    pub fn from_index(&self, index: usize) -> (M::Measure, M) {
        // Bounds check
        if let Some(out) = self.get_from_index(index) {
            out
        } else {
            panic!(
                "Attempt to index past end of slice: index {}, slice length {}",
                index,
                self.len()
            );
        }
    }

    /// Returns the [`M`][Measurable] at `measure` and the starting measure sum
    /// of that element.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `measure` is out of bounds (i.e. `measure >
    /// RopeSlice::measure()`).
    #[inline]
    pub fn from_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> (M::Measure, M) {
        if let Some(out) = self.get_from_measure(measure, cmp) {
            out
        } else {
            panic!(
                "Attempt to index past end of slice: measure {:?}, slice measure {:?}",
                measure,
                self.measure()
            );
        }
    }

    /// Returns the chunk containing the given index.
    ///
    /// Also returns the index and measure of the beginning of the chunk.
    ///
    /// Note: for convenience, a one-past-the-end `index` returns the last
    /// chunk of the [`RopeSlice<M>`].
    ///
    /// The return value is organized as `(chunk, chunk_index, chunk_measure)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `index` is out of bounds (i.e. `index >
    /// RopeSlice::len()`).
    #[inline]
    pub fn chunk_at_index(&self, index: usize) -> (&'a [M], usize, M::Measure) {
        self.try_chunk_at_index(index).unwrap()
    }

    /// Returns the chunk containing the given measure.
    ///
    /// Also returns the index and measure of the beginning of the chunk.
    ///
    /// Note: for convenience, a one-past-the-end `measure` returns the last
    /// chunk of the [`RopeSlice<M>`].
    ///
    /// The return value is organized as `(chunk, chunk_index, chunk_measure)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `measure` is out of bounds (i.e. `measure >
    /// RopeSlice::measure()`).
    #[inline]
    pub fn chunk_at_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> (&'a [M], usize, M::Measure) {
        if let Some(out) = self.get_chunk_at_measure(measure, cmp) {
            out
        } else {
            panic!(
                "Attempt to index past end of slice: measure {:?}, slice measure {:?}",
                measure,
                self.measure()
            );
        }
    }

    /// Returns the entire contents of the [`RopeSlice<M>`] as a
    /// [`&[M]`][Measurable] if possible.
    ///
    /// This is useful for optimizing cases where the [`RopeSlice<M>`] is not
    /// very long.
    ///
    /// For large slices this method will typically fail and return [`None`]
    /// because large slices usually cross chunk boundaries in the rope.
    ///
    /// (Also see the [`From`] impl for converting to a
    /// [`Cow<&[M]>`][std::borrow::Cow].)
    ///
    /// Runs in O(1) time.
    #[inline]
    pub fn as_slice(&self) -> Option<&'a [M]> {
        match *self {
            RopeSlice(RSEnum::Full { .. }) => None,
            RopeSlice(RSEnum::Light { slice }) => Some(slice),
        }
    }

    //-----------------------------------------------------------------------
    // Slice creation

    /// Gets an sub-slice of the [`RopeSlice<M>`], using a measure range.
    ///
    /// Uses range syntax, e.g. `2..7`, `2..`, etc.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use any_rope::Rope;
    /// # use any_rope::Width;
    /// let mut rope = Rope::from_slice(&[
    ///     Width(1),
    ///     Width(2),
    ///     Width(3),
    ///     Width(0),
    ///     Width(0),
    ///     Width(2),
    ///     Width(1),
    /// ]);
    /// let slice = rope.measure_slice(..5, usize::cmp);
    ///
    /// assert_eq!(slice, [Width(1), Width(2), Width(3)].as_slice());
    /// ```
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end, or if the
    /// end is out of bounds (i.e. `end > RopeSlice::measure()`).
    #[inline]
    pub fn measure_slice(
        &self,
        range: impl MeasureRange<M>,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Self {
        let (start, end) = measures_from_range(&range, self.measure()).unwrap();

        if cmp(&start, &M::Measure::default()).is_eq() && cmp(&end, &self.measure()).is_eq() {
            return *self;
        }

        match *self {
            RopeSlice(RSEnum::Full { node, .. }) => {
                RopeSlice::new_with_range(node, start, end, &cmp)
            }
            RopeSlice(RSEnum::Light { slice, .. }) => {
                let start_index = start_measure_to_index(slice, start, &cmp);
                let end_index = end_measure_to_index(slice, end, cmp);
                let new_slice = &slice[start_index..end_index];
                RopeSlice(RSEnum::Light { slice: new_slice })
            }
        }
    }

    /// Gets and sub-slice of the [`RopeSlice<M>`], using an index range.
    ///
    /// Uses range syntax, e.g. `2..7`, `2..`, etc.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The start of the range is greater than the end.
    /// - The end is out of bounds (i.e. `end > Rope::len()`).
    #[inline]
    pub fn index_slice<R>(&self, index_range: R) -> Self
    where
        R: RangeBounds<usize>,
    {
        match self.get_slice_impl(index_range) {
            Ok(s) => return s,
            Err(e) => panic!("slice(): {}", e),
        }
    }

    //-----------------------------------------------------------------------
    // Iterator methods

    /// Creates an iterator over the [`RopeSlice<M>`].
    ///
    /// This iterator will return values of type [`Option<(usize, M)>`], where
    /// the `usize` is the measure sum where the given [`M`][Measurable]
    /// starts.
    ///
    /// Runs in O(log N) time.
    #[inline]
    pub fn iter(&self) -> Iter<'a, M> {
        match *self {
            RopeSlice(RSEnum::Full {
                node,
                start_info,
                end_info,
            }) => Iter::new_with_range(
                node,
                (start_info.len as usize, end_info.len as usize),
                (start_info.measure, end_info.measure),
                M::Measure::fallible_cmp,
            ),
            RopeSlice(RSEnum::Light { slice, .. }) => Iter::from_slice(slice),
        }
    }

    /// Creates an iterator over the [`RopeSlice<M>`], starting at `measure`.
    ///
    /// This iterator will return values of type [`Option<(usize, M)>`], where
    /// the `usize` is the measure where the given [`M`][Measurable] starts.
    /// Since one can iterate in between an [`M`][Measurable]s start and end
    /// measure sums. the first `usize` may not actually corelate to the
    /// `measure` given to the function.
    ///
    /// If `measure == RopeSlice::measure()` then an iterator at the end of the
    /// [`RopeSlice<M>`] is created (i.e. [`next()`][crate::iter::Iter::next]
    /// will return [`None`]).
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `measure` is out of bounds (i.e. `measure >
    /// RopeSlice::measure()`).
    #[inline]
    pub fn iter_at(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Iter<'a, M> {
        if let Some(out) = self.get_iter_at(measure, cmp) {
            out
        } else {
            panic!(
                "Attempt to index past end of RopeSlice: measure {:?}, RopeSlice measure {:?}",
                measure,
                self.measure()
            );
        }
    }

    /// Creates an iterator over the chunks of the [`RopeSlice<M>`].
    ///
    /// Runs in O(log N) time.
    #[inline]
    pub fn chunks(&self) -> Chunks<'a, M> {
        match *self {
            RopeSlice(RSEnum::Full {
                node,
                start_info,
                end_info,
            }) => Chunks::new_with_range(
                node,
                (start_info.len as usize, end_info.len as usize),
                (start_info.measure, end_info.measure),
            ),
            RopeSlice(RSEnum::Light { slice, .. }) => Chunks::from_slice(slice, false),
        }
    }

    /// Creates an iterator over the chunks of the [`RopeSlice<M>`], with the
    /// iterator starting at the chunk containing the `index`.
    ///
    /// Also returns the index and measure of the beginning of the first
    /// chunk to be yielded.
    ///
    /// If `index == RopeSlice::len()` an iterator at the end of the
    /// [`RopeSlice<M>`] (yielding [`None`] on a call to
    /// [`next()`][crate::iter::Iter::next]) is created.
    ///
    /// The return value is organized as `(iterator, chunk_index,
    /// chunk_measure)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `index` is out of bounds (i.e. `index >
    /// RopeSlice::len()`).
    #[inline]
    pub fn chunks_at_index(&self, index: usize) -> (Chunks<'a, M>, usize, M::Measure) {
        if let Some(out) = self.get_chunks_at_index(index) {
            out
        } else {
            panic!(
                "Attempt to index past end of RopeSlice: index {}, RopeSlice length {}",
                index,
                self.len()
            );
        }
    }

    /// Creates an iterator over the chunks of the [`RopeSlice<M>`], with the
    /// iterator starting at the chunk containing the `measure`.
    ///
    /// Also returns the index and measure of the beginning of the first
    /// chunk to be yielded.
    ///
    /// If `measure == RopeSlice::measure()` an iterator at the end of the
    /// [`RopeSlice<M>`] (yielding [`None`] on a call to
    /// [`next()`][crate::iter::Iter::next]) is created.
    ///
    /// The return value is organized as `(iterator, chunk_index,
    /// chunk_measure)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `measure` is out of bounds (i.e. `measure >
    /// RopeSlice::measure()`).
    #[inline]
    pub fn chunks_at_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> (Chunks<'a, M>, usize, M::Measure) {
        if let Some(out) = self.get_chunks_at_measure(measure, cmp) {
            out
        } else {
            panic!(
                "Attempt to index past end of RopeSlice: measure {:?}, RopeSlice measure {:?}",
                measure,
                self.measure()
            );
        }
    }
}

/// # Non-Panicking
///
/// The methods in this impl block provide non-panicking versions of
/// [`RopeSlice<M>`]'s panicking methods. They return either `Option::None` or
/// `Result::Err()` when their panicking counterparts would have panicked.
impl<'a, M> RopeSlice<'a, M>
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    /// Non-panicking version of
    /// [`index_to_measure()`][RopeSlice::index_to_measure].
    #[inline]
    pub fn try_index_to_measure(&self, index: usize) -> Result<M::Measure, M> {
        // Bounds check
        if index <= self.len() {
            let (chunk, b, c) = self.chunk_at_index(index);
            Ok(c + index_to_measure(chunk, index - b))
        } else {
            Err(Error::IndexOutOfBounds(index, self.len()))
        }
    }

    /// Non-panicking version of
    /// [`start_measure_to_index()`][RopeSlice::start_measure_to_index].
    #[inline]
    pub fn try_start_measure_to_index(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Result<usize, M> {
        // Bounds check
        if cmp(&measure, &self.measure()).is_le() {
            let (chunk, b, c) = self.chunk_at_measure(measure, &cmp);
            Ok(b + start_measure_to_index(chunk, measure - c, cmp))
        } else {
            Err(Error::MeasureOutOfBounds(measure, self.measure()))
        }
    }

    /// Non-panicking version of
    /// [`end_measure_to_index()`][RopeSlice::end_measure_to_index].
    #[inline]
    pub fn try_end_measure_to_index(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Result<usize, M> {
        // Bounds check
        if cmp(&measure, &self.measure()).is_le() {
            let (chunk, b, c) = self.chunk_at_measure(measure, &cmp);
            Ok(b + end_measure_to_index(chunk, measure - c, cmp))
        } else {
            Err(Error::MeasureOutOfBounds(measure, self.measure()))
        }
    }

    /// Non-panicking version of [`from_index()`][RopeSlice::from_index].
    #[inline]
    pub fn get_from_index(&self, index: usize) -> Option<(M::Measure, M)> {
        // Bounds check
        if index < self.len() {
            let (chunk, chunk_index, chunk_measure) = self.chunk_at_index(index);
            let chunk_rel_index = index - chunk_index;
            let measure = index_to_measure(chunk, chunk_rel_index);
            Some((measure + chunk_measure, chunk[chunk_rel_index]))
        } else {
            None
        }
    }

    /// Non-panicking version of [`from_measure()`][RopeSlice::from_measure].
    #[inline]
    pub fn get_from_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Option<(M::Measure, M)> {
        // Bounds check
        if cmp(&measure, &self.measure()).is_lt() {
            let (chunk, _, chunk_measure) = self.chunk_at_measure(measure, &cmp);
            let index = start_measure_to_index(chunk, measure - chunk_measure, cmp);
            let measure = index_to_measure(chunk, index);
            Some((measure + chunk_measure, chunk[index]))
        } else {
            None
        }
    }

    /// Non-panicking version of
    /// [`chunk_at_index()`][RopeSlice::chunk_at_index].
    #[inline]
    pub fn try_chunk_at_index(&self, index: usize) -> Result<(&'a [M], usize, M::Measure), M> {
        // Bounds check
        if index <= self.len() {
            match *self {
                RopeSlice(RSEnum::Full {
                    node,
                    start_info,
                    end_info,
                }) => {
                    // Get the chunk.
                    let (chunk, chunk_start_info) =
                        node.get_chunk_at_index(index + start_info.len as usize);

                    // Calculate clipped start/end indices within the chunk.
                    let chunk_start_index = start_info.len.saturating_sub(chunk_start_info.len);
                    let chunk_end_index =
                        (chunk.len() as Count).min(end_info.len - chunk_start_info.len);

                    // Return the clipped chunk and index offset.
                    Ok((
                        &chunk[chunk_start_index as usize..chunk_end_index as usize],
                        chunk_start_info.len.saturating_sub(start_info.len) as usize,
                        fallible_saturating_sub(chunk_start_info.measure, start_info.measure),
                    ))
                }
                RopeSlice(RSEnum::Light { slice, .. }) => Ok((slice, 0, M::Measure::default())),
            }
        } else {
            Err(Error::IndexOutOfBounds(index, self.len()))
        }
    }

    /// Non-panicking version of
    /// [`chunk_at_measure()`][RopeSlice::chunk_at_measure].
    #[inline]
    pub fn get_chunk_at_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Option<(&'a [M], usize, M::Measure)> {
        // Bounds check
        if cmp(&measure, &self.measure()).is_le() {
            match *self {
                RopeSlice(RSEnum::Full {
                    node,
                    start_info,
                    end_info,
                }) => {
                    let start_measure = measure + start_info.measure;
                    let (chunk, chunk_start_info) =
                        node.get_first_chunk_at_measure(start_measure, &cmp);

                    // Calculate clipped start/end indices within the chunk.
                    let chunk_start_index = start_info.len.saturating_sub(chunk_start_info.len);
                    let chunk_end_index =
                        (chunk.len() as Count).min(end_info.len - chunk_start_info.len);

                    // Return the clipped chunk and index offset.
                    Some((
                        &chunk[chunk_start_index as usize..chunk_end_index as usize],
                        chunk_start_info.len.saturating_sub(start_info.len) as usize,
                        fallible_saturating_sub(chunk_start_info.measure, start_info.measure),
                    ))
                }
                RopeSlice(RSEnum::Light { slice, .. }) => Some((slice, 0, M::Measure::default())),
            }
        } else {
            None
        }
    }

    /// Non-panicking version of [`measure_slice()`][RopeSlice::measure_slice].
    #[inline]
    pub fn get_measure_slice(
        &self,
        range: impl MeasureRange<M>,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Option<RopeSlice<'a, M>> {
        let (start, end) = measures_from_range(&range, self.measure()).unwrap();

        if cmp(&start, &M::Measure::default()).is_eq() && cmp(&end, &self.measure()).is_eq() {
            return Some(*self);
        }

        // Bounds check
        if cmp(&start, &end).is_le() && cmp(&end, &self.measure()).is_le() {
            match *self {
                RopeSlice(RSEnum::Full {
                    node, start_info, ..
                }) => Some(RopeSlice::new_with_range(
                    node,
                    start_info.measure + start,
                    start_info.measure + end,
                    &cmp,
                )),
                RopeSlice(RSEnum::Light { slice, .. }) => {
                    let start_index = start_measure_to_index(slice, start, &cmp);
                    let end_index = start_measure_to_index(slice, end, cmp);
                    let new_slice = &slice[start_index..end_index];
                    Some(RopeSlice(RSEnum::Light { slice: new_slice }))
                }
            }
        } else {
            None
        }
    }

    /// Non-panicking version of [`index_slice()`][RopeSlice::index_slice].
    #[inline]
    pub fn get_index_slice<R>(&self, index_range: R) -> Option<RopeSlice<'a, M>>
    where
        R: RangeBounds<usize>,
    {
        self.get_slice_impl(index_range).ok()
    }

    pub(crate) fn get_slice_impl<R>(&self, index_range: R) -> Result<RopeSlice<'a, M>, M>
    where
        R: RangeBounds<usize>,
    {
        let start_range = start_bound_to_num(index_range.start_bound());
        let end_range = end_bound_to_num(index_range.end_bound());

        // Bounds checks.
        match (start_range, end_range) {
            (None, None) => {
                // Early-out shortcut for taking a slice of the full thing.
                return Ok(*self);
            }
            (Some(s), Some(e)) => {
                if s > e {
                    return Err(Error::IndexRangeInvalid(s, e));
                } else if e > self.len() {
                    return Err(Error::IndexRangeOutOfBounds(
                        start_range,
                        end_range,
                        self.len(),
                    ));
                }
            }
            (Some(s), None) => {
                if s > self.len() {
                    return Err(Error::IndexRangeOutOfBounds(
                        start_range,
                        end_range,
                        self.len(),
                    ));
                }
            }
            (None, Some(e)) => {
                if e > self.len() {
                    return Err(Error::IndexRangeOutOfBounds(
                        start_range,
                        end_range,
                        self.len(),
                    ));
                }
            }
        }

        let (start, end) = (
            start_range.unwrap_or(0),
            end_range.unwrap_or_else(|| self.len()),
        );

        match *self {
            RopeSlice(RSEnum::Full {
                node, start_info, ..
            }) => RopeSlice::new_with_index_range(
                node,
                start_info.len as usize + start,
                start_info.len as usize + end,
            ),
            RopeSlice(RSEnum::Light { slice, .. }) => {
                let new_slice = &slice[start..end];
                Ok(RopeSlice(RSEnum::Light { slice: new_slice }))
            }
        }
    }

    /// Non-panicking version of [`iter_at()`][RopeSlice::iter_at].
    #[inline]
    pub fn get_iter_at(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Option<Iter<'a, M>> {
        // Bounds check
        if cmp(&measure, &self.measure()).is_le() {
            match *self {
                RopeSlice(RSEnum::Full {
                    node,
                    start_info,
                    end_info,
                }) => Some(Iter::new_with_range_at_measure(
                    node,
                    start_info.measure + measure,
                    (start_info.len as usize, end_info.len as usize),
                    (start_info.measure, end_info.measure),
                    cmp,
                )),
                RopeSlice(RSEnum::Light { slice, .. }) => {
                    Some(Iter::from_slice_at(slice, measure, &cmp))
                }
            }
        } else {
            None
        }
    }

    /// Non-panicking version of
    /// [`chunks_at_index()`][RopeSlice::chunks_at_index].
    #[inline]
    pub fn get_chunks_at_index(&self, index: usize) -> Option<(Chunks<'a, M>, usize, M::Measure)> {
        // Bounds check
        if index <= self.len() {
            match *self {
                RopeSlice(RSEnum::Full {
                    node,
                    start_info,
                    end_info,
                }) => {
                    let (chunks, chunk_index, chunk_measure) = Chunks::new_with_range_at_index(
                        node,
                        index + start_info.len as usize,
                        (start_info.len as usize, end_info.len as usize),
                        (start_info.measure, end_info.measure),
                    );

                    Some((
                        chunks,
                        chunk_index.saturating_sub(start_info.len as usize),
                        chunk_measure - start_info.measure,
                    ))
                }
                RopeSlice(RSEnum::Light { slice }) => {
                    let chunks = Chunks::from_slice(slice, index == slice.len());

                    if index == slice.len() {
                        Some((chunks, slice.len(), measure_of(slice)))
                    } else {
                        Some((chunks, 0, M::Measure::default()))
                    }
                }
            }
        } else {
            None
        }
    }

    /// Non-panicking version of
    /// [`chunks_at_measure()`][RopeSlice::chunks_at_measure].
    #[inline]
    pub fn get_chunks_at_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Option<(Chunks<'a, M>, usize, M::Measure)> {
        // Bounds check
        if cmp(&measure, &self.measure()).is_le() {
            match *self {
                RopeSlice(RSEnum::Full {
                    node,
                    start_info,
                    end_info,
                }) => {
                    let (chunks, chunk_index, chunk_measure) = Chunks::new_with_range_at_measure(
                        node,
                        measure + start_info.measure,
                        (start_info.len as usize, end_info.len as usize),
                        (start_info.measure, end_info.measure),
                        cmp,
                    );

                    Some((
                        chunks,
                        chunk_index.saturating_sub(start_info.len as usize),
                        chunk_measure - start_info.measure,
                    ))
                }
                RopeSlice(RSEnum::Light { slice, .. }) => {
                    let slice_measure = measure_of(slice);
                    let chunks = Chunks::from_slice(slice, cmp(&measure, &slice_measure).is_eq());

                    if cmp(&measure, &slice_measure).is_eq() {
                        Some((chunks, slice.len(), slice_measure))
                    } else {
                        Some((chunks, 0, M::Measure::default()))
                    }
                }
            }
        } else {
            None
        }
    }
}

//==============================================================
// Conversion impls

/// Creates a [`RopeSlice<M>`] directly from a [`&[M]`][Measurable] slice.
///
/// The useful applications of this are actually somewhat narrow. It is
/// intended primarily as an aid when implementing additional functionality
/// on top of AnyRope, where you may already have access to a rope chunk and
/// want to directly create a [`RopeSlice<M>`] from it, avoiding the overhead of
/// going through the slicing APIs.
///
/// Although it is possible to use this to create [`RopeSlice<M>`]s from
/// arbitrary lists, doing so is not especially useful. For example,
/// [`Rope<M>`]s and [`RopeSlice<M>`]s can already be directly compared for
/// equality with [`Vec<M>`] and [`&[M]`][Measurable] slices.
///
/// Runs in O(N) time, where N is the length of the slice.
impl<'a, M> From<&'a [M]> for RopeSlice<'a, M>
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn from(slice: &'a [M]) -> Self {
        RopeSlice(RSEnum::Light { slice })
    }
}

impl<'a, M> From<RopeSlice<'a, M>> for Vec<M>
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn from(s: RopeSlice<'a, M>) -> Self {
        let mut vec = Vec::with_capacity(s.len());
        vec.extend(s.chunks().flat_map(|chunk| chunk.iter()).copied());
        vec
    }
}

/// Attempts to borrow the contents of the slice, but will convert to an
/// owned [`Vec<M>`] if the contents is not contiguous in memory.
///
/// Runs in best case O(1), worst case O(N).
impl<'a, M> From<RopeSlice<'a, M>> for std::borrow::Cow<'a, [M]>
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn from(s: RopeSlice<'a, M>) -> Self {
        if let Some(slice) = s.as_slice() {
            std::borrow::Cow::Borrowed(slice)
        } else {
            std::borrow::Cow::Owned(Vec::from(s))
        }
    }
}

//==============================================================
// Other impls

impl<'a, M> std::fmt::Debug for RopeSlice<'a, M>
where
    M: Measurable + std::fmt::Debug,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_list().entries(self.chunks()).finish()
    }
}

impl<'a, M> std::fmt::Display for RopeSlice<'a, M>
where
    M: Measurable + std::fmt::Display,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("[")?;

        let mut iter = self.iter();
        iter.next()
            .map(|(_, measurable)| f.write_fmt(format_args!("{}", measurable)))
            .transpose()?;

        for (_, measurable) in iter {
            f.write_fmt(format_args!(", {}", measurable))?;
        }
        f.write_str("]")
    }
}

impl<'a, M> std::cmp::Eq for RopeSlice<'a, M>
where
    M: Measurable + Eq,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
}

impl<'a, 'b, M> std::cmp::PartialEq<RopeSlice<'b, M>> for RopeSlice<'a, M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    fn eq(&self, other: &RopeSlice<'b, M>) -> bool {
        if self.len() != other.len() {
            return false;
        }

        let mut chunk_itr_1 = self.chunks();
        let mut chunk_itr_2 = other.chunks();
        let mut chunk1 = chunk_itr_1.next().unwrap_or(&[]);
        let mut chunk2 = chunk_itr_2.next().unwrap_or(&[]);

        loop {
            if chunk1.len() > chunk2.len() {
                if &chunk1[..chunk2.len()] != chunk2 {
                    return false;
                } else {
                    chunk1 = &chunk1[chunk2.len()..];
                    chunk2 = &[];
                }
            } else if &chunk2[..chunk1.len()] != chunk1 {
                return false;
            } else {
                chunk2 = &chunk2[chunk1.len()..];
                chunk1 = &[];
            }

            if chunk1.is_empty() {
                if let Some(chunk) = chunk_itr_1.next() {
                    chunk1 = chunk;
                } else {
                    break;
                }
            }

            if chunk2.is_empty() {
                if let Some(chunk) = chunk_itr_2.next() {
                    chunk2 = chunk;
                } else {
                    break;
                }
            }
        }

        return true;
    }
}

impl<'a, 'b, M> std::cmp::PartialEq<&'b [M]> for RopeSlice<'a, M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &&'b [M]) -> bool {
        match *self {
            RopeSlice(RSEnum::Full { .. }) => {
                if self.len() != other.len() {
                    return false;
                }

                let mut index = 0;
                for chunk in self.chunks() {
                    if chunk != &other[index..(index + chunk.len())] {
                        return false;
                    }
                    index += chunk.len();
                }

                return true;
            }
            RopeSlice(RSEnum::Light { slice, .. }) => {
                return slice == *other;
            }
        }
    }
}

impl<'a, 'b, M> std::cmp::PartialEq<RopeSlice<'a, M>> for &'b [M]
where
    M: Measurable + PartialEq,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &RopeSlice<'a, M>) -> bool {
        other == self
    }
}

impl<'a, M> std::cmp::PartialEq<[M]> for RopeSlice<'a, M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &[M]) -> bool {
        std::cmp::PartialEq::<&[M]>::eq(self, &other)
    }
}

impl<'a, M> std::cmp::PartialEq<RopeSlice<'a, M>> for [M]
where
    M: Measurable + PartialEq,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &RopeSlice<'a, M>) -> bool {
        std::cmp::PartialEq::<&[M]>::eq(other, &self)
    }
}

impl<'a, M> std::cmp::PartialEq<Vec<M>> for RopeSlice<'a, M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &Vec<M>) -> bool {
        self == other.as_slice()
    }
}

impl<'a, M> std::cmp::PartialEq<RopeSlice<'a, M>> for Vec<M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &RopeSlice<'a, M>) -> bool {
        self.as_slice() == other
    }
}

impl<'a, 'b, M> std::cmp::PartialEq<std::borrow::Cow<'b, [M]>> for RopeSlice<'a, M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &std::borrow::Cow<'b, [M]>) -> bool {
        *self == **other
    }
}

impl<'a, 'b, M> std::cmp::PartialEq<RopeSlice<'a, M>> for std::borrow::Cow<'b, [M]>
where
    M: Measurable + PartialEq,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &RopeSlice<'a, M>) -> bool {
        **self == *other
    }
}

impl<'a, M> std::cmp::PartialEq<Rope<M>> for RopeSlice<'a, M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        *self == other.measure_slice(.., M::Measure::fallible_cmp)
    }
}

impl<'a, M> std::cmp::PartialEq<RopeSlice<'a, M>> for Rope<M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &RopeSlice<'a, M>) -> bool {
        self.measure_slice(.., M::Measure::fallible_cmp) == *other
    }
}

impl<'a, M> std::cmp::Ord for RopeSlice<'a, M>
where
    M: Measurable + Ord,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[allow(clippy::op_ref)] // Erroneously thinks with can directly use a slice.
    fn cmp(&self, other: &RopeSlice<'a, M>) -> std::cmp::Ordering {
        let mut chunk_itr_1 = self.chunks();
        let mut chunk_itr_2 = other.chunks();
        let mut chunk1 = chunk_itr_1.next().unwrap_or(&[]);
        let mut chunk2 = chunk_itr_2.next().unwrap_or(&[]);

        loop {
            if chunk1.len() >= chunk2.len() {
                let compared = chunk1[..chunk2.len()].cmp(chunk2);
                if compared != std::cmp::Ordering::Equal {
                    return compared;
                }

                chunk1 = &chunk1[chunk2.len()..];
                chunk2 = &[];
            } else {
                let compared = chunk1.cmp(&chunk2[..chunk1.len()]);
                if compared != std::cmp::Ordering::Equal {
                    return compared;
                }

                chunk1 = &[];
                chunk2 = &chunk2[chunk1.len()..];
            }

            if chunk1.is_empty() {
                if let Some(chunk) = chunk_itr_1.next() {
                    chunk1 = chunk;
                } else {
                    break;
                }
            }

            if chunk2.is_empty() {
                if let Some(chunk) = chunk_itr_2.next() {
                    chunk2 = chunk;
                } else {
                    break;
                }
            }
        }

        self.len().cmp(&other.len())
    }
}

impl<'a, 'b, M> std::cmp::PartialOrd<RopeSlice<'b, M>> for RopeSlice<'a, M>
where
    M: Measurable + PartialOrd + Ord,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    #[inline]
    fn partial_cmp(&self, other: &RopeSlice<'b, M>) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

//===========================================================

#[cfg(test)]
mod tests {
    use crate::{
        slice_utils::{index_to_measure, start_measure_to_index},
        Rope, Width,
    };

    /// 70 elements, total measure of 135.
    fn pseudo_random() -> Vec<Width> {
        (0..70)
            .map(|num| match num % 14 {
                0 | 7 => Width(1),
                1 | 8 => Width(2),
                2 => Width(4),
                3 | 10 => Width(0),
                4 | 11 => Width(0),
                5 => Width(5),
                6 => Width(1),
                9 => Width(8),
                12 => Width(3),
                13 => Width(0),
                _ => unreachable!(),
            })
            .collect()
    }

    #[test]
    fn len_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(7..30, usize::cmp);
        assert_eq!(slice.len(), 13);
    }

    #[test]
    fn len_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(43..43, usize::cmp);
        assert_eq!(slice.len(), 0);
    }

    #[test]
    fn measure_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(7..30, usize::cmp);
        assert_eq!(slice.measure(), 23);
    }

    #[test]
    fn measure_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(43..43, usize::cmp);
        assert_eq!(slice.measure(), 0);
    }

    #[test]
    fn measure_03() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(6..30, usize::cmp);
        assert_eq!(slice.measure(), 24);
    }

    #[test]
    fn index_to_measure_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(88.., usize::cmp);

        assert_eq!(slice.index_to_measure(0), 0); // Width(0): 0
        assert_eq!(slice.index_to_measure(1), 0); // Width(0): 0
        assert_eq!(slice.index_to_measure(2), 0); // Width("hello"): 5
        assert_eq!(slice.index_to_measure(3), 5); // Width(true): 1
        assert_eq!(slice.index_to_measure(4), 6); // Width(1): 1
        assert_eq!(slice.index_to_measure(5), 7); // Width(2): 2
        assert_eq!(slice.index_to_measure(6), 9); // Width(8): 8
        assert_eq!(slice.index_to_measure(7), 17); // Width(0): 0
        assert_eq!(slice.index_to_measure(8), 17); // Width(0): 0
        assert_eq!(slice.index_to_measure(9), 17); // Width("bye"): 3
        assert_eq!(slice.index_to_measure(10), 20); // Width(false): 0
        assert_eq!(slice.index_to_measure(11), 20); // Width(0): 1
        assert_eq!(slice.index_to_measure(12), 21); // Width(0): 2
        assert_eq!(slice.index_to_measure(13), 23); // Width(4): 4
    }

    #[test]
    fn measure_to_index_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(88..135, usize::cmp);

        // NOTE: For some elements, it may seem like the amount of "measures"
        // that correspond to their index may not match their actual measure.
        // For example, `Width("bye")` only lasts for 2 measures, even
        // though its measure is 3.
        // This is because there are 0 measure elements preceding it, and the
        // measure in `measure_to_index()` merely corresponds to the end of an
        // element, and has no relation to the referred element's measure.
        assert_eq!(slice.start_measure_to_index(0, usize::cmp), 0);
        assert_eq!(slice.start_measure_to_index(1, usize::cmp), 2);
        assert_eq!(slice.start_measure_to_index(4, usize::cmp), 2);
        assert_eq!(slice.start_measure_to_index(5, usize::cmp), 3);
        assert_eq!(slice.start_measure_to_index(6, usize::cmp), 4);
        assert_eq!(slice.start_measure_to_index(7, usize::cmp), 5);
        assert_eq!(slice.start_measure_to_index(8, usize::cmp), 5);
        assert_eq!(slice.start_measure_to_index(9, usize::cmp), 6);
        assert_eq!(slice.start_measure_to_index(16, usize::cmp), 6);
        assert_eq!(slice.start_measure_to_index(17, usize::cmp), 7);
        assert_eq!(slice.start_measure_to_index(18, usize::cmp), 9);
        assert_eq!(slice.start_measure_to_index(19, usize::cmp), 9);
        assert_eq!(slice.start_measure_to_index(20, usize::cmp), 10);
        assert_eq!(slice.start_measure_to_index(21, usize::cmp), 12);
        assert_eq!(slice.start_measure_to_index(22, usize::cmp), 12);
        assert_eq!(slice.start_measure_to_index(23, usize::cmp), 13);
        assert_eq!(slice.start_measure_to_index(24, usize::cmp), 13);
    }

    #[test]
    fn from_index_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(34..100, usize::cmp);

        assert_eq!(slice.from_index(0), (0, Width(0)));
        assert_eq!(slice.from_index(10), (20, Width(0)));

        assert_eq!(slice.from_index(slice.len() - 3), (60, Width(1)));
        assert_eq!(slice.from_index(slice.len() - 2), (61, Width(2)));
        assert_eq!(slice.from_index(slice.len() - 1), (63, Width(8)));
    }

    #[test]
    #[should_panic]
    fn from_index_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(34..100, usize::cmp);
        slice.from_index(slice.len());
    }

    #[test]
    #[should_panic]
    fn from_index_03() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(42..42, usize::cmp);
        slice.from_index(0);
    }

    #[test]
    fn from_measure_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(34..100, usize::cmp);

        assert_eq!(slice.from_measure(0, usize::cmp), (0, Width(0)));
        assert_eq!(slice.from_measure(3, usize::cmp), (0, Width(5)));
        assert_eq!(slice.from_measure(5, usize::cmp), (5, Width(1)));
        assert_eq!(slice.from_measure(6, usize::cmp), (6, Width(1)));
        assert_eq!(slice.from_measure(10, usize::cmp), (9, Width(8)));
        assert_eq!(slice.from_measure(65, usize::cmp), (63, Width(8)));
    }

    #[test]
    #[should_panic]
    fn from_measure_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(34..100, usize::cmp);
        slice.from_measure(66, usize::cmp);
    }

    #[test]
    #[should_panic]
    fn from_measure_03() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(43..43, usize::cmp);
        slice.from_measure(0, usize::cmp);
    }

    #[test]
    fn chunk_at_index_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice_1 = rope.measure_slice(34..135, usize::cmp);
        let slice_2 = &pseudo_random()[17..70];
        // "'slice a fine day, isn't it?\nAren't you glad \
        //  we're alive?\nこんにちは、みん"

        let mut total = slice_2;
        let mut prev_chunk = [].as_slice();
        for i in 0..slice_1.len() {
            let (chunk, index, measure) = slice_1.chunk_at_index(i);
            assert_eq!(measure, index_to_measure(slice_2, index));
            if chunk != prev_chunk {
                assert_eq!(chunk, &total[..chunk.len()]);
                total = &total[chunk.len()..];
                prev_chunk = chunk;
            }

            let measure_1 = slice_2.get(i).unwrap();
            let measure_2 = {
                let i2 = i - index;
                chunk.get(i2).unwrap()
            };
            assert_eq!(measure_1, measure_2);
        }

        assert_eq!(total.len(), 0);
    }

    #[test]
    fn chunk_at_measure_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice_1 = rope.measure_slice(34..96, usize::cmp);
        let slice_2 = &pseudo_random()[17..51];

        let mut remainder = slice_2;
        let mut prev_chunk = [].as_slice();

        for i in 0..slice_1.measure() {
            let (chunk, _, measure) = slice_1.chunk_at_measure(i, usize::cmp);
            if chunk != prev_chunk {
                assert_eq!(chunk, &remainder[..chunk.len()]);
                remainder = &remainder[chunk.len()..];
                prev_chunk = chunk;
            }

            let measure_1 = {
                let index_1 = start_measure_to_index(slice_2, i, usize::cmp);
                slice_2.get(index_1).unwrap()
            };
            let measure_2 = {
                let index_2 = start_measure_to_index(chunk, i - measure, usize::cmp);
                chunk.get(index_2)
            };
            if let Some(measure_2) = measure_2 {
                assert_eq!(measure_1, measure_2);
            }
        }
        assert_eq!(remainder.len(), 0);
    }

    #[test]
    fn slice_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice_1 = rope.measure_slice(.., usize::cmp);

        let slice_2 = slice_1.measure_slice(.., usize::cmp);

        assert_eq!(pseudo_random(), slice_2);
    }

    #[test]
    fn slice_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice_1 = rope.measure_slice(5..43, usize::cmp);

        let slice_2 = slice_1.measure_slice(3..25, usize::cmp);

        assert_eq!(&pseudo_random()[2..13], slice_2);
    }

    #[test]
    fn slice_03() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice_1 = rope.measure_slice(31..97, usize::cmp);

        let slice_2 = slice_1.measure_slice(7..64, usize::cmp);

        assert_eq!(&pseudo_random()[17..48], slice_2);
    }

    #[test]
    fn slice_04() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice_1 = rope.measure_slice(5..43, usize::cmp);

        // A range in the middle of a list of 0 elements should capture all of
        // them.
        let slice_2 = slice_1.measure_slice(24..24, usize::cmp);

        assert_eq!(slice_2, [Width(0), Width(0)].as_slice());
    }

    #[test]
    #[should_panic]
    fn slice_05() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(5..43, usize::cmp);

        #[allow(clippy::reversed_empty_ranges)]
        slice.measure_slice(21..20, usize::cmp); // Wrong ordering on purpose.
    }

    #[test]
    #[should_panic]
    fn slice_07() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(5..43, usize::cmp);

        slice.measure_slice(37..39, usize::cmp);
    }

    #[test]
    fn eq_slice_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(.., usize::cmp);

        assert_eq!(slice, pseudo_random());
        assert_eq!(pseudo_random(), slice);
    }

    #[test]
    fn eq_slice_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(0..20, usize::cmp);

        assert_ne!(slice, pseudo_random());
        assert_ne!(pseudo_random(), slice);
    }

    #[test]
    fn eq_slice_03() {
        let mut rope = Rope::from_slice(pseudo_random().as_slice());
        rope.remove_inclusive(20..20, usize::cmp);
        rope.insert_slice(20, &[Width(5)], usize::cmp);
        let slice = rope.measure_slice(.., usize::cmp);

        assert_ne!(slice, pseudo_random());
        assert_ne!(pseudo_random(), slice);
    }

    #[test]
    fn eq_slice_04() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(.., usize::cmp);
        let vec: Vec<Width> = rope.clone().into();

        assert_eq!(slice, vec);
        assert_eq!(vec, slice);
    }

    #[test]
    fn eq_rope_slice_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(43..43, usize::cmp);

        assert_eq!(slice, slice);
    }

    #[test]
    fn eq_rope_slice_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice_1 = rope.measure_slice(43..97, usize::cmp);
        let slice_2 = rope.measure_slice(43..97, usize::cmp);

        assert_eq!(slice_1, slice_2);
    }

    #[test]
    fn eq_rope_slice_03() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice_1 = rope.measure_slice(43..43, usize::cmp);
        let slice_2 = rope.measure_slice(43..45, usize::cmp);

        assert_ne!(slice_1, slice_2);
    }

    #[test]
    fn eq_rope_slice_04() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice_1 = rope.measure_slice(43..45, usize::cmp);
        let slice_2 = rope.measure_slice(43..43, usize::cmp);

        assert_ne!(slice_1, slice_2);
    }

    #[test]
    fn eq_rope_slice_05() {
        let rope: Rope<Width> = Rope::from_slice([].as_slice());
        let slice = rope.measure_slice(0..0, usize::cmp);

        assert_eq!(slice, slice);
    }

    #[test]
    fn cmp_rope_slice_01() {
        let rope_1 = Rope::from_slice(pseudo_random().as_slice());
        let rope_2 = Rope::from_slice(pseudo_random().as_slice());
        let slice_1 = rope_1.measure_slice(.., usize::cmp);
        let slice_2 = rope_2.measure_slice(.., usize::cmp);

        assert_eq!(slice_1.cmp(&slice_2), std::cmp::Ordering::Equal);
        assert_eq!(
            slice_1.measure_slice(..24, usize::cmp).cmp(&slice_2),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            slice_1.cmp(&slice_2.measure_slice(..24, usize::cmp)),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn cmp_rope_slice_02() {
        let rope_1 = Rope::from_slice(&[Width(3), Width(1), Width(2), Width(1)]);
        let rope_2 = Rope::from_slice(&[Width(3), Width(0), Width(2), Width(1)]);
        let slice_1 = rope_1.measure_slice(.., usize::cmp);
        let slice_2 = rope_2.measure_slice(.., usize::cmp);

        assert_eq!(slice_1.cmp(&slice_2), std::cmp::Ordering::Greater);
        assert_eq!(slice_2.cmp(&slice_1), std::cmp::Ordering::Less);
    }

    #[test]
    fn to_vec_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(.., usize::cmp);
        let vec: Vec<Width> = slice.into();

        assert_eq!(rope, vec);
        assert_eq!(vec, slice);
    }

    #[test]
    fn to_vec_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(0..24, usize::cmp);
        let vec: Vec<Width> = slice.into();

        assert_eq!(slice, vec);
    }

    #[test]
    fn to_vec_03() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(13..89, usize::cmp);
        let vec: Vec<Width> = slice.into();

        assert_eq!(slice, vec);
    }

    #[test]
    fn to_vec_04() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(13..41, usize::cmp);
        let vec: Vec<Width> = slice.into();

        assert_eq!(slice, vec);
    }

    #[test]
    fn to_cow_01() {
        use std::borrow::Cow;
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(13..83, usize::cmp);
        let cow: Cow<[Width]> = slice.into();

        assert_eq!(slice, cow);
    }

    #[test]
    fn to_cow_02() {
        use std::borrow::Cow;
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.measure_slice(13..14, usize::cmp);
        let cow: Cow<[Width]> = rope.measure_slice(13..14, usize::cmp).into();

        // Make sure it's borrowed.
        if let Cow::Owned(_) = cow {
            panic!("Small Cow conversions should result in a borrow.");
        }

        assert_eq!(slice, cow);
    }
}
