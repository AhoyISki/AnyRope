use std::{
    arch::x86_64::_CMP_EQ_US, cmp::Ordering, iter::FromIterator, ops::RangeBounds, sync::Arc,
};

use crate::{
    end_bound_to_num,
    iter::{Chunks, Iter},
    measures_from_range,
    rope_builder::RopeBuilder,
    slice::RopeSlice,
    slice_utils::{end_measure_to_index, index_to_measure, start_measure_to_index},
    start_bound_to_num,
    tree::{max_children, max_len, min_len, BranchChildren, Node, SliceInfo},
    Error, Measurable, MeasureRange, Result,
};

/// A rope of elements that are [`Measurable`].
///
/// The time complexity of nearly all edit and query operations on [`Rope<M>`]
/// are worst-case `O(log N)` in the length of the rope. [`Rope<M>`] is designed
/// to work efficiently even for huge (in the gigabytes) arrays of
/// [`M`][Measurable].
///
/// In the examples below, a struct called [`Width`][crate::Width] will be used
/// in order to demonstrate AnyRope's features.
///
/// # Editing Operations
///
/// The primary editing operations on [`Rope<M>`] are insertion and removal of
/// slices or individual elements.
/// For example:
///
/// ```
/// # use any_rope::{Rope, Width};
/// let mut rope = Rope::from_slice(&[
///     Width(1),
///     Width(2),
///     Width(3),
///     Width(0),
///     Width(0),
///     Width(2),
///     Width(1),
/// ]);
/// rope.remove_inclusive(6..8);
/// rope.insert(6, Width(5));
///
/// assert_eq!(
///     rope,
///     [Width(1), Width(2), Width(3), Width(5), Width(1)].as_slice()
/// );
/// ```
///
/// # Query Operations
///
/// [`Rope<M>`] gives you the ability to query an element at any given index or
/// measure, and the convertion between the two. You can either convert an index
/// to a measure, or convert the measure at the start or end of an element to an
/// index. For example:
///
/// ```
/// # use any_rope::{Rope, Width};
/// let rope = Rope::from_slice(&[
///     Width(0),
///     Width(0),
///     Width(1),
///     Width(1),
///     Width(2),
///     Width(25),
///     Width(0),
///     Width(0),
///     Width(1),
/// ]);
///
/// // `start_measure_to_index()` will pick the first element that starts at the given index.
/// assert_eq!(rope.start_measure_to_index(0), 0);
/// // `end_measure_to_index()` will pick the last element that still starts at the given index.
/// assert_eq!(rope.end_measure_to_index(0), 2);
/// assert_eq!(rope.start_measure_to_index(2), 4);
/// assert_eq!(rope.start_measure_to_index(3), 4);
/// assert_eq!(rope.start_measure_to_index(16), 5);
/// assert_eq!(rope.start_measure_to_index(29), 6);
/// ```
///
/// # Slicing
///
/// You can take immutable slices of a [`Rope<M>`] using
/// [`measure_slice()`][Rope::measure_slice]
/// or [`index_slice()`][Rope::index_slice]:
///
/// ```
/// # use any_rope::{Rope, Width};
/// let mut rope = Rope::from_slice(&[
///     Width(1),
///     Width(2),
///     Width(3),
///     Width(0),
///     Width(0),
///     Width(2),
///     Width(1),
/// ]);
/// let measure_slice = rope.measure_slice(3..6);
/// let index_slice = rope.index_slice(2..5);
///
/// assert_eq!(measure_slice, index_slice);
/// ```
///
/// # Cloning
///
/// Cloning [`Rope<M>`]s is extremely cheap, running in `O(1)` time and taking a
/// small constant amount of memory for the new clone, regardless of slice size.
/// This is accomplished by data sharing between [`Rope<M>`] clones. The memory
/// used by clones only grows incrementally as the their contents diverge due
/// to edits. All of this is thread safe, so clones can be sent freely
/// between threads.
///
/// The primary intended use-case for this feature is to allow asynchronous
/// processing of [`Rope<M>`]s.
#[derive(Clone)]
pub struct Rope<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    pub(crate) root: Arc<Node<M>>,
}

impl<M> Rope<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    //-----------------------------------------------------------------------
    // Constructors

    /// Creates an empty [`Rope<M>`].
    #[inline]
    pub fn new() -> Self {
        Rope {
            root: Arc::new(Node::new()),
        }
    }

    /// Creates a [`Rope<M>`] from an [`M`][Measurable] slice.
    ///
    /// Runs in O(N) time.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn from_slice(slice: &[M]) -> Self {
        RopeBuilder::new().build_at_once(slice)
    }

    //-----------------------------------------------------------------------
    // Informational methods

    /// Total number of elements in [`Rope<M>`].
    ///
    /// Runs in O(N) time.
    #[inline]
    pub fn len(&self) -> usize {
        self.root.len()
    }

    /// Returns `true` if the [`Rope<M>`] is empty.
    ///
    /// Runs in O(N) time.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.root.len() == 0
    }

    /// Sum of all measures of in [`Rope<M>`].
    ///
    /// Runs in O(N) time.
    #[inline]
    pub fn measure(&self) -> M::Measure {
        self.root.measure()
    }

    //-----------------------------------------------------------------------
    // Memory management methods

    /// Total size of the [`Rope<M>`]'s buffer space.
    ///
    /// This includes unoccupied buffer space. You can calculate
    /// the unoccupied space with `Rope::capacity() - Rope::len()`. In general,
    /// there will always be some unoccupied buffer space.
    ///
    /// Runs in O(N) time.
    #[inline]
    pub fn capacity(&self) -> usize {
        let mut count = 0;
        for chunk in self.chunks() {
            count += chunk.len().max(max_len::<M>());
        }
        count
    }

    /// Shrinks the [`Rope<M>`]'s capacity to the minimum possible.
    ///
    /// This will rarely result in `Rope::capacity() == Rope::len()`.
    /// [`Rope<M>`] stores [`M`][Measurable]s in a sequence of
    /// fixed-capacity chunks, so an exact fit only happens for lists of a
    /// lenght that is a multiple of that capacity.
    ///
    /// After calling this, the difference between `capacity()` and
    /// `len()` is typically under 1000 for each 1000000 [`M`][Measurable] in
    /// the [`Rope<M>`].
    ///
    /// **NOTE:** calling this on a [`Rope<M>`] clone causes it to stop sharing
    /// all data with its other clones. In such cases you will very likely
    /// be _increasing_ total memory usage despite shrinking the [`Rope<M>`]'s
    /// capacity.
    ///
    /// Runs in O(N) time, and uses O(log N) additional space during
    /// shrinking.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        let mut node_stack = Vec::new();
        let mut builder = RopeBuilder::new();

        node_stack.push(self.root.clone());
        *self = Rope::new();

        loop {
            if node_stack.is_empty() {
                break;
            }

            if node_stack.last().unwrap().is_leaf() {
                builder.append_slice(node_stack.pop().unwrap().leaf_slice());
            } else if node_stack.last().unwrap().child_count() == 0 {
                node_stack.pop();
            } else {
                let (_, next_node) = Arc::make_mut(node_stack.last_mut().unwrap())
                    .children_mut()
                    .remove(0);
                node_stack.push(next_node);
            }
        }

        *self = builder.finish();
    }

    //-----------------------------------------------------------------------
    // Edit methods

    /// Inserts [`slice`][Measurable] at `measure`.
    ///
    /// Runs in O(L + log N) time, where N is the length of the [`Rope<M>`] and
    /// L is the length of [`slice`][Measurable].
    ///
    /// # Panics
    ///
    /// Panics if the `measure` is out of bounds (i.e. `measure >
    /// Rope::measure()`).
    #[inline]
    pub fn insert_slice(
        &mut self,
        measure: M::Measure,
        slice: &[M],
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) {
        // Bounds check
        self.try_insert_slice(measure, slice, cmp).unwrap()
    }

    /// Inserts a single [`M`][Measurable] at `measure`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `measure` is out of bounds (i.e. `measure >
    /// Rope::measure()`).
    #[inline]
    pub fn insert(&mut self, measure: M::Measure, measurable: M) {
        self.try_insert(measure, measurable).unwrap()
    }

    /// Private internal-only method that does a single insertion of
    /// a sufficiently small slice.
    ///
    /// This only works correctly for insertion slices smaller than or equal to
    /// `MAX_BYTES - 4`.
    #[inline]
    fn insert_internal(
        &mut self,
        measure: M::Measure,
        slice: &[M],
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) {
        let root_info = self.root.info();

        let (l_info, residual) = Arc::make_mut(&mut self.root).edit_chunk_at_measure(
            measure,
            root_info,
            |index, cur_info, leaf_slice| {
                // Find our index
                let index = end_measure_to_index(leaf_slice, index, &cmp);

                // No node splitting
                if (leaf_slice.len() + slice.len()) <= max_len::<M>() {
                    // Calculate new info without doing a full re-scan of cur_slice.
                    let new_info = cur_info + SliceInfo::from_slice(slice);
                    leaf_slice.insert_slice(index, slice);
                    (new_info, None)
                }
                // We're splitting the node
                else {
                    let r_slice = leaf_slice.insert_slice_split(index, slice);
                    let l_slice_info = SliceInfo::from_slice(leaf_slice);
                    if r_slice.len() > 0 {
                        let r_slice_info = SliceInfo::from_slice(&r_slice);
                        (
                            l_slice_info,
                            Some((r_slice_info, Arc::new(Node::Leaf(r_slice)))),
                        )
                    } else {
                        // Leaf couldn't be validly split, so leave it oversized
                        (l_slice_info, None)
                    }
                }
            },
        );

        // Handle root splitting, if any.
        if let Some((r_info, r_node)) = residual {
            let mut l_node = Arc::new(Node::new());
            std::mem::swap(&mut l_node, &mut self.root);

            let mut children = BranchChildren::new();
            children.push((l_info, l_node));
            children.push((r_info, r_node));

            *Arc::make_mut(&mut self.root) = Node::Branch(children);
        }
    }

    /// Removes the slice in the given measure range.
    ///
    /// Uses range syntax, e.g. `2..7`, `2..`, etc.
    ///
    /// Runs in O(M + log N) time, where N is the length of the [`Rope<M>`] and
    /// M is the length of the range being removed.
    ///
    /// The first removed [`M`][Measurable] will be the first with a end measure
    /// sum greater than the starting bound if its
    /// [`measure()`][Measurable::measure] is greater than 0, or equal to the
    /// starting bound if its [`measure()`][Measurable::measure] is equal to 0.
    ///
    /// The last removed [`M`][Measurable] will be the first with a start
    /// measure sum greater than the ending bound if its
    /// [`measure()`][Measurable::measure] is greater than 0, or the last one
    /// with a start measure sum equal to the ending bound if its
    /// [`measure()`][Measurable::measure] is equal to 0.
    ///
    /// In essence, this means the following:
    /// - A range starting between a [`M`][Measurable]'s start and end measure
    ///   sums will remove
    /// said [`M`][Measurable].
    /// - A range ending in the start of a list of 0 measure [`M`][Measurable]s
    ///   will remove
    /// all of them.
    /// - An empty range that starts and ends in a list of 0 measure
    ///   [`M`][Measurable]s will
    /// remove all of them, and nothing else. This contrasts with Rust's usual
    /// definition of an empty range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use any_rope::{Rope, Width};
    /// let array = [
    ///     Width(1),
    ///     Width(2),
    ///     Width(3),
    ///     Width(0),
    ///     Width(0),
    ///     Width(2),
    ///     Width(1),
    /// ];
    /// let mut rope = Rope::from_slice(&array);
    ///
    /// // Removing in the middle of `Width(3)`.
    /// rope.remove_inclusive(5..);
    ///
    /// assert_eq!(rope, [Width(1), Width(2)].as_slice());
    /// ```
    /// ```rust
    /// # use any_rope::{Rope, Width};
    /// let array = [
    ///     Width(1),
    ///     Width(2),
    ///     Width(3),
    ///     Width(0),
    ///     Width(0),
    ///     Width(2),
    ///     Width(1),
    /// ];
    /// let mut rope = Rope::from_slice(&array);
    ///
    /// // End bound coincides with a 0 measure list.
    /// rope.remove_inclusive(1..6);
    ///
    /// assert_eq!(rope, [Width(1), Width(2), Width(1)].as_slice());
    /// ```
    /// ```rust
    /// # use any_rope::{Rope, Width};
    /// let array = [
    ///     Width(1),
    ///     Width(2),
    ///     Width(3),
    ///     Width(0),
    ///     Width(0),
    ///     Width(2),
    ///     Width(1),
    /// ];
    /// let mut rope = Rope::from_slice(&array);
    ///
    /// // Empty range at the start of a 0 measure list.
    /// rope.remove_inclusive(6..6);
    ///
    /// // Inclusively removing an empty range does nothing.
    /// assert_eq!(
    ///     rope,
    ///     [Width(1), Width(2), Width(3), Width(2), Width(1)].as_slice()
    /// );
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end, or if the
    /// end is out of bounds (i.e. `end > self.measure()`).
    #[inline]
    pub fn remove_inclusive(
        &mut self,
        range: impl MeasureRange<M>,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) {
        self.try_remove_inclusive(range, cmp).unwrap()
    }

    /// Same as [`remove_inclusive()`][Rope::remove_inclusive], but keeps
    /// elements measure measure equal to 0 at the edges.
    ///
    /// If the `measure_range` doesn't cover the entire measure of a single
    /// [`M`][Measurable], then the removal does nothing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use any_rope::{Rope, Width};
    /// let array = [
    ///     Width(1),
    ///     Width(2),
    ///     Width(3),
    ///     Width(0),
    ///     Width(0),
    ///     Width(2),
    ///     Width(1),
    /// ];
    /// let mut rope = Rope::from_slice(&array);
    ///
    /// // End bound coincides with a 0 measure list, which does not get removed.
    /// rope.remove_exclusive(1..6);
    ///
    /// assert_eq!(
    ///     rope,
    ///     [Width(1), Width(0), Width(0), Width(2), Width(1)].as_slice()
    /// );
    /// ```
    /// ```rust
    /// # use any_rope::{Rope, Width};
    /// let array = [
    ///     Width(1),
    ///     Width(2),
    ///     Width(3),
    ///     Width(0),
    ///     Width(0),
    ///     Width(2),
    ///     Width(1),
    /// ];
    /// let mut rope = Rope::from_slice(&array);
    ///
    /// // Empty range at the start of a 0 measure list.
    /// rope.remove_exclusive(6..6);
    ///
    /// // Exclusively removing an empty range does nothing.
    /// assert_eq!(rope, array.as_slice());
    /// ```
    /// ```rust
    /// # use any_rope::{Rope, Width};
    /// let array = [
    ///     Width(1),
    ///     Width(2),
    ///     Width(3),
    ///     Width(0),
    ///     Width(0),
    ///     Width(2),
    ///     Width(1),
    /// ];
    /// let mut rope = Rope::from_slice(&array);
    ///
    /// // Removing in the middle of `Width(3)`.
    /// rope.remove_exclusive(5..6);
    ///
    /// // Exclusively removing in the middle of an element does nothing.
    /// assert_eq!(rope, array.as_slice());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end, or if the
    /// end is out of bounds (i.e. `end > self.measure()`).
    #[inline]
    pub fn remove_exclusive(
        &mut self,
        range: impl MeasureRange<M>,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) {
        self.try_remove_exclusive(range, cmp).unwrap()
    }

    /// Splits the [`Rope<M>`] at `measure`, returning the right part of the
    /// split.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `measure` is out of bounds (i.e. `measure >
    /// self.measure()`).
    #[inline]
    pub fn split_off(
        &mut self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Self {
        self.try_split_off(measure, cmp).unwrap()
    }

    /// Appends a [`Rope<M>`] to the end of this one, consuming the other
    /// [`Rope<M>`].
    ///
    /// Runs in O(log N) time.
    #[inline]
    pub fn append(&mut self, mut other: Self) {
        if self.measure() == M::Measure::default() {
            // Special case
            std::mem::swap(self, &mut other);
        } else if other.measure() > M::Measure::default() {
            let left_info = self.root.info();
            let right_info = other.root.info();

            let l_depth = self.root.depth();
            let r_depth = other.root.depth();

            if l_depth > r_depth {
                let extra =
                    Arc::make_mut(&mut self.root).append_at_depth(other.root, l_depth - r_depth);
                if let Some(node) = extra {
                    let mut children = BranchChildren::new();
                    children.push((self.root.info(), Arc::clone(&self.root)));
                    children.push((node.info(), node));
                    self.root = Arc::new(Node::Branch(children));
                }
            } else {
                let mut other = other;
                let extra = Arc::make_mut(&mut other.root)
                    .prepend_at_depth(Arc::clone(&self.root), r_depth - l_depth);
                if let Some(node) = extra {
                    let mut children = BranchChildren::new();
                    children.push((node.info(), node));
                    children.push((other.root.info(), Arc::clone(&other.root)));
                    other.root = Arc::new(Node::Branch(children));
                }
                *self = other;
            };

            // Fix up any mess left behind.
            let root = Arc::make_mut(&mut self.root);
            if (left_info.len as usize) < min_len::<M>()
                || (right_info.len as usize) < min_len::<M>()
            {
                root.fix_tree_seam(left_info.measure, &M::Measure::cmp);
            }
            self.pull_up_singular_nodes();
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
    /// Rope::measure()`).
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
    /// Rope::measure()`).
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
    /// Panics if the `index` is out of bounds (i.e. `index > Rope::len()`).
    #[inline]
    pub fn from_index(&self, index: usize) -> (M::Measure, M) {
        // Bounds check
        if let Some(out) = self.get_from_index(index) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: index {}, Rope length {}",
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
    /// Rope::measure()`).
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
                "Attempt to index past end of Rope: measure {:?}, Total rope measure is {:?}",
                measure,
                self.measure()
            );
        }
    }

    /// Returns the chunk containing the given index.
    ///
    /// Also returns the index and widht of the beginning of the chunk.
    ///
    /// Note: for convenience, a one-past-the-end `index` returns the last
    /// chunk of the `RopeSlice`.
    ///
    /// The return value is organized as
    /// `(chunk, chunk_index, chunk_measure)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `index` is out of bounds (i.e. `index > Rope::len()`).
    #[inline]
    pub fn chunk_at_index(&self, index: usize) -> (&[M], usize, M::Measure) {
        // Bounds check
        if let Some(out) = self.get_chunk_at_index(index) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: index {}, Rope length {}",
                index,
                self.len()
            );
        }
    }

    /// Returns the chunk containing the given measure.
    ///
    /// Also returns the index and measure of the beginning of the chunk.
    ///
    /// Note: for convenience, a one-past-the-end `measure` returns the last
    /// chunk of the `RopeSlice`.
    ///
    /// The return value is organized as
    /// `(chunk, chunk_index, chunk_measure)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `measure` is out of bounds (i.e. `measure >
    /// Rope::measure()`).
    #[inline]
    pub fn chunk_at_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> (&[M], usize, M::Measure) {
        if let Some(out) = self.get_chunk_at_measure(measure, &cmp) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: measure {:?}, Rope measure {:?}",
                measure,
                self.measure()
            );
        }
    }

    //-----------------------------------------------------------------------
    // Slicing

    /// Gets an immutable slice of the [`Rope<M>`], using a measure range.
    ///
    /// Uses range syntax, e.g. `2..7`, `2..`, etc.
    ///
    /// # Example
    ///
    /// ```
    /// # use any_rope::{Rope, Width};
    /// let mut rope = Rope::from_slice(&[
    ///     Width(1),
    ///     Width(2),
    ///     Width(3),
    ///     Width(0),
    ///     Width(0),
    ///     Width(2),
    ///     Width(1),
    /// ]);
    /// let slice = rope.measure_slice(..5);
    ///
    /// assert_eq!(slice, [Width(1), Width(2), Width(3)].as_slice());
    /// ```
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end, or if the
    /// end is out of bounds (i.e. `end > Rope::measure()`).
    #[inline]
    pub fn measure_slice(
        &self,
        measure_range: impl MeasureRange<M>,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> RopeSlice<M> {
        self.get_measure_slice(measure_range, cmp).unwrap()
    }

    /// Gets and immutable slice of the [`Rope<M>`], using an index range.
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
    pub fn index_slice(&self, index_range: impl RangeBounds<usize>) -> RopeSlice<M> {
        match self.get_index_slice_impl(index_range) {
            Ok(s) => return s,
            Err(e) => panic!("index_slice(): {}", e),
        }
    }

    //-----------------------------------------------------------------------
    // Iterator methods

    /// Creates an iterator over the [`Rope<M>`].
    ///
    /// This iterator will return values of type [Option<(usize, M)>], where the
    /// `usize` is the measure sum where the given [`M`][Measurable] starts.
    ///
    /// Runs in O(log N) time.
    #[inline]
    pub fn iter(&self) -> Iter<M> {
        Iter::new(&self.root)
    }

    /// Creates an iterator over the  [`Rope<M>`], starting at `measure`.
    ///
    /// This iterator will return values of type [`Option<(usize, M)>`], where
    /// the `usize` is the measure where the given [`M`][Measurable] starts.
    /// Since one can iterate in between an [`M`][Measurable]s start and end
    /// measure sums. the first `usize` may not actually corelate to the
    /// `measure` given to the function.
    ///
    /// If `measure == Rope::measure()` then an iterator at the end of the
    /// [`Rope<M>`] is created (i.e. [`next()`][crate::iter::Iter::next] will
    /// return [`None`]).
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `measure` is out of bounds (i.e. `measure >
    /// Rope::measure()`).
    #[inline]
    pub fn iter_at_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Iter<M> {
        if let Some(out) = self.get_iter_at_measure(measure, cmp) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: measure {:?}, Rope measure {:?}",
                measure,
                self.measure()
            );
        }
    }

    /// Creates an iterator over the chunks of the [`Rope<M>`].
    ///
    /// Runs in O(log N) time.
    #[inline]
    pub fn chunks(&self) -> Chunks<M> {
        Chunks::new(&self.root)
    }

    /// Creates an iterator over the chunks of the [`Rope<M>`], with the
    /// iterator starting at the chunk containing the `index`.
    ///
    /// Also returns the index and measure of the beginning of the first
    /// chunk to be yielded.
    ///
    /// If `index == Rope::len()` an iterator at the end of the [`Rope<M>`]
    /// (yielding [`None`] on a call to [`next()`][crate::iter::Iter::next]) is
    /// created.
    ///
    /// The return value is organized as `(iterator, chunk_index,
    /// chunk_measure)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the `index` is out of bounds (i.e. `index > Rope::len()`).
    #[inline]
    pub fn chunks_at_index(&self, index: usize) -> (Chunks<M>, usize, M::Measure) {
        if let Some(out) = self.get_chunks_at_index(index) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: index {}, Rope length {}",
                index,
                self.len()
            );
        }
    }

    /// Creates an iterator over the chunks of the [`Rope<M>`], with the
    /// iterator starting at the chunk containing the `measure`.
    ///
    /// Also returns the index and measure of the beginning of the first
    /// chunk to be yielded.
    ///
    /// If `measure == Rope::measure()` an iterator at the end of the
    /// [`Rope<M>`] (yielding [`None`] on a call to
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
    /// Rope::measure()`).
    #[inline]
    pub fn chunks_at_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> (Chunks<M>, usize, M::Measure) {
        if let Some(out) = self.get_chunks_at_measure(measure, &cmp) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: measure {:?}, Rope measure {:?}",
                measure,
                self.measure()
            );
        }
    }

    /// Returns true if this rope and `other` point to precisely the same
    /// in-memory data.
    ///
    /// This happens when one of the ropes is a clone of the other and
    /// neither have been modified since then. Because clones initially
    /// share all the same data, it can be useful to check if they still
    /// point to precisely the same memory as a way of determining
    /// whether they are both still unmodified.
    ///
    /// Note: this is distinct from checking for equality: two ropes can
    /// have the same *contents* (equal) but be stored in different
    /// memory locations (not instances). Importantly, two clones that
    /// post-cloning are modified identically will *not* be instances
    /// anymore, even though they will have equal contents.
    ///
    /// Runs in O(1) time.
    #[inline]
    pub fn is_instance(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.root, &other.root)
    }

    //-----------------------------------------------------------------------
    // Debugging

    /// NOT PART OF THE PUBLIC API (hidden from docs for a reason!)
    ///
    /// Debugging tool to make sure that all of the meta-data of the
    /// tree is consistent with the actual data.
    #[doc(hidden)]
    pub fn assert_integrity(&self) {
        self.root.assert_integrity();
    }

    /// NOT PART OF THE PUBLIC API (hidden from docs for a reason!)
    ///
    /// Debugging tool to make sure that all of the following invariants
    /// hold true throughout the tree:
    ///
    /// - The tree is the same height everywhere.
    /// - All internal nodes have the minimum number of children.
    /// - All leaf nodes are non-empty.
    #[doc(hidden)]
    pub fn assert_invariants(&self) {
        self.root.assert_balance();
        self.root.assert_node_size(true);
    }

    //-----------------------------------------------------------------------
    // Internal utilities

    /// Iteratively replaces the root node with its child if it only has
    /// one child.
    #[inline]
    pub(crate) fn pull_up_singular_nodes(&mut self) {
        while (!self.root.is_leaf()) && self.root.child_count() == 1 {
            let child = if let Node::Branch(ref children) = *self.root {
                Arc::clone(&children.nodes()[0])
            } else {
                unreachable!()
            };

            self.root = child;
        }
    }
}

/// # Non-Panicking
///
/// The methods in this impl block provide non-panicking versions of
/// [`Rope<M>`]'s panicking methods. They return either `Option::None` or
/// `Result::Err()` when their panicking counterparts would have panicked.
impl<M> Rope<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    /// Non-panicking version of [`insert()`][Rope::insert].
    #[inline]
    pub fn try_insert_slice(
        &mut self,
        measure: M::Measure,
        mut slice: &[M],
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Result<(), M> {
        // Bounds check
        if measure <= self.measure() {
            // We have three cases here:
            // 1. The insertion slice is very large, in which case building a new Rope out
            //    of it and splicing it into the existing Rope is most efficient.
            // 2. The insertion slice is somewhat large, in which case splitting it up into
            //    chunks and repeatedly inserting them is the most efficient. The splitting
            //    is necessary because the insertion code only works correctly below a
            //    certain insertion size.
            // 3. The insertion slice is small, in which case we can simply insert it.
            //
            // Cases #2 and #3 are rolled into one case here, where case #3 just
            // results in the slice being "split" into only one chunk.
            //
            // The boundary for what constitutes "very large" slice was arrived at
            // experimentally, by testing at what point Rope build + splice becomes
            // faster than split + repeated insert.
            if slice.len() > max_len::<M>() * 6 {
                // Case #1: very large slice, build rope and splice it in.
                let rope = Rope::from_slice(slice);
                let right = self.split_off(measure, &cmp);
                self.append(rope);
                self.append(right);
            } else {
                // Cases #2 and #3: split into chunks and repeatedly insert.
                while !slice.is_empty() {
                    // Split a chunk off from the end of the slice.
                    // We do this from the end instead of the front so that
                    // the repeated insertions can keep re-using the same
                    // insertion point.
                    let split_index = slice.len() - (max_len::<M>() - 4).min(slice.len());
                    let ins_slice = &slice[split_index..];
                    slice = &slice[..split_index];

                    // Do the insertion.
                    self.insert_internal(measure, ins_slice, M::Measure::cmp);
                }
            }
            Ok(())
        } else {
            Err(Error::MeasureOutOfBounds(measure, self.measure()))
        }
    }

    /// Non-panicking version of [`insert()`][Rope::insert].
    #[inline]
    pub fn try_insert(&mut self, measure: M::Measure, measurable: M) -> Result<(), M> {
        // Bounds check
        if measure <= self.measure() {
            self.insert_internal(measure, &[measurable], M::Measure::cmp);
            Ok(())
        } else {
            Err(Error::MeasureOutOfBounds(measure, self.measure()))
        }
    }

    /// Non-panicking version of [`remove_inclusive()`][Rope::remove_inclusive].
    #[inline]
    pub fn try_remove_inclusive(
        &mut self,
        range: impl MeasureRange<M>,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Result<(), M> {
        self.try_remove_internal(range, cmp, true)
    }

    /// Non-panicking version of [`remove_exclusive()`][Rope::remove_exclusive].
    #[inline]
    pub fn try_remove_exclusive(
        &mut self,
        range: impl MeasureRange<M>,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Result<(), M> {
        self.try_remove_internal(range, cmp, false)
    }

    fn try_remove_internal(
        &mut self,
        range: impl MeasureRange<M>,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
        inclusive: bool,
    ) -> Result<(), M> {
        let (start, end) = measures_from_range(&range, self.measure())?;

        if cmp(&start, &M::Measure::default()).is_eq()
            && cmp(&end, &self.measure()).is_eq()
            && inclusive
        {
            self.root = Arc::new(Node::new());
            Ok(())
        } else {
            let root = Arc::make_mut(&mut self.root);

            let (_, needs_fix) = root.remove_range(start, end, &cmp, inclusive, inclusive);

            if needs_fix {
                root.fix_tree_seam(start, &cmp);
            }

            self.pull_up_singular_nodes();
            Ok(())
        }
    }

    /// Non-panicking version of [`split_off()`][Rope::split_off].
    #[inline]
    pub fn try_split_off(
        &mut self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Result<Self, M> {
        // Bounds check
        if measure <= self.measure() {
            if measure == M::Measure::default() {
                // Special case 1
                let mut new_rope = Rope::new();
                std::mem::swap(self, &mut new_rope);
                Ok(new_rope)
            } else if measure == self.measure() {
                // Special case 2
                Ok(Rope::new())
            } else {
                // Do the split
                let mut new_rope = Rope {
                    root: Arc::new(Arc::make_mut(&mut self.root).end_split(measure, cmp)),
                };

                // Fix up the edges
                Arc::make_mut(&mut self.root).zip_fix_right();
                Arc::make_mut(&mut new_rope.root).zip_fix_left();
                self.pull_up_singular_nodes();
                new_rope.pull_up_singular_nodes();

                Ok(new_rope)
            }
        } else {
            Err(Error::MeasureOutOfBounds(measure, self.measure()))
        }
    }

    /// Non-panicking version of [`index_to_measure()`][Rope::index_to_measure].
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
    /// [`start_measure_to_index()`][Rope::start_measure_to_index].
    #[inline]
    pub fn try_start_measure_to_index(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Result<usize, M> {
        // Bounds check
        if measure <= self.measure() {
            let (chunk, b, c) = self.chunk_at_measure(measure, &cmp);
            Ok(b + start_measure_to_index(chunk, measure - c, cmp))
        } else {
            Err(Error::MeasureOutOfBounds(measure, self.measure()))
        }
    }

    /// Non-panicking version of
    /// [`end_measure_to_index()`][Rope::end_measure_to_index].
    #[inline]
    pub fn try_end_measure_to_index(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Result<usize, M> {
        // Bounds check
        if measure <= self.measure() {
            let (chunk, b, c) = self.chunk_at_measure(measure, &cmp);
            Ok(b + end_measure_to_index(chunk, measure - c, cmp))
        } else {
            Err(Error::MeasureOutOfBounds(measure, self.measure()))
        }
    }

    /// Non-panicking version of [`from_index()`][Rope::from_index].
    #[inline]
    pub fn get_from_index(&self, index: usize) -> Option<(M::Measure, M)> {
        // Bounds check
        if index < self.len() {
            let (chunk, chunk_index, chunk_measure) = self.chunk_at_index(index);
            let index = index - chunk_index;
            let measure = index_to_measure(chunk, index);
            Some((measure + chunk_measure, chunk[index]))
        } else {
            None
        }
    }

    /// Non-panicking version of [`from_measure()`][Rope::from_measure].
    #[inline]
    pub fn get_from_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Option<(M::Measure, M)> {
        // Bounds check
        if measure <= self.measure() && !self.is_empty() {
            let (chunk, _, chunk_measure) = self.chunk_at_measure(measure, &cmp);
            let index = start_measure_to_index(chunk, measure - chunk_measure, cmp);
            let measure = index_to_measure(chunk, index);
            Some((measure + chunk_measure, chunk[index.min(chunk.len() - 1)]))
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunk_at_index()`][Rope::chunk_at_index].
    #[inline]
    pub fn get_chunk_at_index(&self, index: usize) -> Option<(&[M], usize, M::Measure)> {
        // Bounds check
        if index <= self.len() {
            let (chunk, info) = self.root.get_chunk_at_index(index);
            Some((chunk, info.len as usize, info.measure))
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunk_at_measure()`][Rope::chunk_at_measure].
    #[inline]
    pub fn get_chunk_at_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Option<(&[M], usize, M::Measure)> {
        // Bounds check
        if measure <= self.measure() && !self.is_empty() {
            let (chunk, info) = self.root.get_first_chunk_at_measure(measure, cmp);
            Some((chunk, info.len as usize, info.measure))
        } else {
            None
        }
    }

    /// Non-panicking version of [`measure_slice()`][Rope::measure_slice].
    #[inline]
    pub fn get_measure_slice(
        &self,
        range: impl MeasureRange<M>,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Result<RopeSlice<M>, M> {
        let measure = self.measure();
        let (start, end) = measures_from_range(&range, self.measure())?;

        // Bounds check
        Ok(RopeSlice::new_with_range(&self.root, start, end, cmp))
    }

    /// Non-panicking version of [`index_slice()`][Rope::index_slice].
    #[inline]
    pub fn get_index_slice<R>(&self, index_range: R) -> Option<RopeSlice<M>>
    where
        R: RangeBounds<usize>,
    {
        self.get_index_slice_impl(index_range).ok()
    }

    pub(crate) fn get_index_slice_impl<R>(&self, index_range: R) -> Result<RopeSlice<M>, M>
    where
        R: RangeBounds<usize>,
    {
        let start_range = start_bound_to_num(index_range.start_bound());
        let end_range = end_bound_to_num(index_range.end_bound());

        // Bounds checks.
        match (start_range, end_range) {
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
                    return Err(Error::IndexRangeOutOfBounds(None, end_range, self.len()));
                }
            }
            _ => {}
        }

        let (start, end) = (
            start_range.unwrap_or(0),
            end_range.unwrap_or_else(|| self.len()),
        );

        RopeSlice::new_with_index_range(&self.root, start, end)
    }

    /// Non-panicking version of [`iter_at_measure()`][Rope::iter_at_measure].
    #[inline]
    pub fn get_iter_at_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Option<Iter<M>> {
        // Bounds check
        if measure <= self.measure() {
            Some(Iter::new_with_range_at_measure(
                &self.root,
                measure,
                (0, self.len()),
                (M::Measure::default(), self.measure()),
                cmp,
            ))
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunks_at_index()`][Rope::chunks_at_index].
    #[inline]
    pub fn get_chunks_at_index(&self, index: usize) -> Option<(Chunks<M>, usize, M::Measure)> {
        // Bounds check
        if index <= self.len() {
            Some(Chunks::new_with_range_at_index(
                &self.root,
                index,
                (0, self.len()),
                (M::Measure::default(), self.measure()),
            ))
        } else {
            None
        }
    }

    /// Non-panicking version of
    /// [`chunks_at_measure()`][Rope::chunks_at_measure].
    #[inline]
    pub fn get_chunks_at_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Option<(Chunks<M>, usize, M::Measure)> {
        // Bounds check
        if measure <= self.measure() {
            Some(Chunks::new_with_range_at_measure(
                &self.root,
                measure,
                (0, self.len()),
                (M::Measure::default(), self.measure()),
                cmp,
            ))
        } else {
            None
        }
    }
}

//==============================================================
// Conversion impls

impl<'a, M> From<&'a [M]> for Rope<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn from(slice: &'a [M]) -> Self {
        Rope::from_slice(slice)
    }
}

impl<'a, M> From<std::borrow::Cow<'a, [M]>> for Rope<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn from(slice: std::borrow::Cow<'a, [M]>) -> Self {
        Rope::from_slice(&slice)
    }
}

impl<M> From<Vec<M>> for Rope<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn from(slice: Vec<M>) -> Self {
        Rope::from_slice(&slice)
    }
}

/// Will share data where possible.
///
/// Runs in O(log N) time.
impl<'a, M> From<RopeSlice<'a, M>> for Rope<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    fn from(s: RopeSlice<'a, M>) -> Self {
        use crate::slice::RSEnum;
        match s {
            RopeSlice(RSEnum::Full {
                node,
                start_info,
                end_info,
            }) => {
                let mut rope = Rope {
                    root: Arc::clone(node),
                };

                // Chop off right end if needed
                if end_info.measure < node.info().measure {
                    {
                        let root = Arc::make_mut(&mut rope.root);
                        root.end_split(end_info.measure, M::Measure::cmp);
                        root.zip_fix_right();
                    }
                    rope.pull_up_singular_nodes();
                }

                // Chop off left end if needed
                if start_info.measure > M::Measure::default() {
                    {
                        let root = Arc::make_mut(&mut rope.root);
                        *root = root.start_split(start_info.measure, M::Measure::cmp);
                        root.zip_fix_left();
                    }
                    rope.pull_up_singular_nodes();
                }

                // Return the rope
                rope
            }
            RopeSlice(RSEnum::Light { slice, .. }) => Rope::from_slice(slice),
        }
    }
}

impl<M> From<Rope<M>> for Vec<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn from(r: Rope<M>) -> Self {
        Vec::from(&r)
    }
}

impl<'a, M> From<&'a Rope<M>> for Vec<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn from(r: &'a Rope<M>) -> Self {
        let mut vec = Vec::with_capacity(r.len());
        vec.extend(r.chunks().flat_map(|chunk| chunk.iter()).copied());
        vec
    }
}

impl<'a, M> From<Rope<M>> for std::borrow::Cow<'a, [M]>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn from(r: Rope<M>) -> Self {
        std::borrow::Cow::Owned(Vec::from(r))
    }
}

/// Attempts to borrow the contents of the [`Rope<M>`], but will convert to an
/// owned [`[M]`][Measurable] if the contents is not contiguous in memory.
///
/// Runs in best case O(1), worst case O(N).
impl<'a, M> From<&'a Rope<M>> for std::borrow::Cow<'a, [M]>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn from(r: &'a Rope<M>) -> Self {
        if let Node::Leaf(ref slice) = *r.root {
            std::borrow::Cow::Borrowed(slice)
        } else {
            std::borrow::Cow::Owned(Vec::from(r))
        }
    }
}

impl<'a, M> FromIterator<&'a [M]> for Rope<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a [M]>,
    {
        let mut builder = RopeBuilder::new();
        for chunk in iter {
            builder.append_slice(chunk);
        }
        builder.finish()
    }
}

impl<'a, M> FromIterator<std::borrow::Cow<'a, [M]>> for Rope<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = std::borrow::Cow<'a, [M]>>,
    {
        let mut builder = RopeBuilder::new();
        for chunk in iter {
            builder.append_slice(&chunk);
        }
        builder.finish()
    }
}

impl<M> FromIterator<Vec<M>> for Rope<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Vec<M>>,
    {
        let mut builder = RopeBuilder::new();
        for chunk in iter {
            builder.append_slice(&chunk);
        }
        builder.finish()
    }
}

//==============================================================
// Other impls

impl<M> std::fmt::Debug for Rope<M>
where
    M: Measurable + std::fmt::Debug,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_list().entries(self.chunks()).finish()
    }
}

impl<M> std::fmt::Display for Rope<M>
where
    M: Measurable + std::fmt::Display,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
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

impl<M> std::default::Default for Rope<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<M> std::cmp::Eq for Rope<M>
where
    M: Measurable + Eq,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
}

impl<M> std::cmp::PartialEq<Rope<M>> for Rope<M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        self.measure_slice(.., M::Measure::cmp) == other.measure_slice(.., M::Measure::cmp)
    }
}

impl<'a, M> std::cmp::PartialEq<&'a [M]> for Rope<M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &&'a [M]) -> bool {
        self.measure_slice(.., M::Measure::cmp) == *other
    }
}

impl<'a, M> std::cmp::PartialEq<Rope<M>> for &'a [M]
where
    M: Measurable + PartialEq,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        *self == other.measure_slice(.., M::Measure::cmp)
    }
}

impl<M> std::cmp::PartialEq<[M]> for Rope<M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &[M]) -> bool {
        self.measure_slice(.., M::Measure::cmp) == other
    }
}

impl<M> std::cmp::PartialEq<Rope<M>> for [M]
where
    M: Measurable + PartialEq,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        self == other.measure_slice(.., M::Measure::cmp)
    }
}

impl<M> std::cmp::PartialEq<Vec<M>> for Rope<M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &Vec<M>) -> bool {
        self.measure_slice(.., M::Measure::cmp) == other.as_slice()
    }
}

impl<M> std::cmp::PartialEq<Rope<M>> for Vec<M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        self.as_slice() == other.measure_slice(.., M::Measure::cmp)
    }
}

impl<'a, M> std::cmp::PartialEq<std::borrow::Cow<'a, [M]>> for Rope<M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &std::borrow::Cow<'a, [M]>) -> bool {
        self.measure_slice(.., M::Measure::cmp) == **other
    }
}

impl<'a, M> std::cmp::PartialEq<Rope<M>> for std::borrow::Cow<'a, [M]>
where
    M: Measurable + PartialEq,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        **self == other.measure_slice(.., M::Measure::cmp)
    }
}

impl<M> std::cmp::Ord for Rope<M>
where
    M: Measurable + Ord,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn cmp(&self, other: &Rope<M>) -> std::cmp::Ordering {
        self.measure_slice(.., M::Measure::cmp).cmp(&other.measure_slice(.., M::Measure::cmp))
    }
}

impl<M> std::cmp::PartialOrd<Rope<M>> for Rope<M>
where
    M: Measurable + PartialOrd + Ord,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline]
    fn partial_cmp(&self, other: &Rope<M>) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

//==============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Width;

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

    /// 5 elements, total measure of 6.
    const SHORT_LOREM: &[Width] = &[Width(1), Width(2), Width(3), Width(0), Width(0)];

    #[test]
    fn new_01() {
        let rope: Rope<Width> = Rope::new();
        assert_eq!(rope, [].as_slice());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn from_slice() {
        let rope = Rope::from(pseudo_random());
        assert_eq!(rope, pseudo_random());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn len_01() {
        let rope = Rope::from(pseudo_random());
        assert_eq!(rope.len(), 70);
    }

    #[test]
    fn measure_02() {
        let rope: Rope<Width> = Rope::from_slice(&[]);
        assert_eq!(rope.len(), 0);
    }

    #[test]
    fn len_from_measures_01() {
        let rope = Rope::from(pseudo_random());
        assert_eq!(rope.measure(), 135);
    }

    #[test]
    fn len_from_measures_02() {
        let rope: Rope<Width> = Rope::from_slice(&[]);
        assert_eq!(rope.measure(), 0);
    }

    #[test]
    fn insert_01() {
        let mut rope = Rope::from_slice(SHORT_LOREM);
        rope.insert_slice(3, &[Width(1), Width(2), Width(3)], usize::cmp);

        assert_eq!(
            rope,
            [
                Width(1),
                Width(2),
                Width(1),
                Width(2),
                Width(3),
                Width(3),
                Width(0),
                Width(0)
            ]
            .as_slice()
        );

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn insert_02() {
        let mut rope = Rope::from_slice(SHORT_LOREM);
        rope.insert_slice(0, &[Width(1), Width(2), Width(3)], usize::cmp);

        assert_eq!(
            rope,
            [
                Width(1),
                Width(2),
                Width(3),
                Width(1),
                Width(2),
                Width(3),
                Width(0),
                Width(0)
            ]
            .as_slice()
        );

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn insert_03() {
        let mut rope = Rope::from_slice(SHORT_LOREM);
        rope.insert_slice(6, &[Width(1), Width(2), Width(3)], usize::cmp);

        assert_eq!(
            rope,
            [
                Width(1),
                Width(2),
                Width(3),
                Width(0),
                Width(0),
                Width(1),
                Width(2),
                Width(3)
            ]
            .as_slice()
        );

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn insert_04() {
        let mut rope = Rope::new();
        rope.insert_slice(0, &[Width(1), Width(2)], usize::cmp);
        rope.insert_slice(2, &[Width(5)], usize::cmp);
        rope.insert_slice(3, &[Width(0)], usize::cmp);
        rope.insert_slice(4, &[Width(4)], usize::cmp);
        rope.insert_slice(11, &[Width(3)], usize::cmp);

        // NOTE: Inserting in the middle of an item'slice measure range, makes it so
        // you actually place it at the end of said item.
        assert_eq!(
            rope,
            [Width(1), Width(2), Width(0), Width(5), Width(4), Width(3)].as_slice()
        );

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn insert_05() {
        let mut rope = Rope::new();
        rope.insert_slice(0, &[Width(15), Width(20)], usize::cmp);
        rope.insert_slice(7, &[Width(0), Width(0)], usize::cmp);
        assert_eq!(rope, [Width(15), Width(0), Width(0), Width(20)].as_slice());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn insert_06() {
        let mut rope = Rope::new();
        rope.insert(0, Width(15));
        rope.insert(1, Width(20));
        rope.insert(2, Width(10));
        rope.insert(3, Width(4));
        rope.insert_slice(20, &[Width(0), Width(0)], usize::cmp);
        assert_eq!(
            rope,
            [
                Width(15),
                Width(4),
                Width(10),
                Width(0),
                Width(0),
                Width(20)
            ]
            .as_slice()
        );

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn remove_01() {
        let slice = &[
            Width(15),
            Width(0),
            Width(0),
            Width(24),
            Width(1),
            Width(2),
            Width(7),
        ];
        let mut rope = Rope::from_slice(slice);

        rope.remove_inclusive(0..11, usize::cmp); // Removes Width(15).
        rope.remove_inclusive(24..31, usize::cmp); // Removes [Width(1), Width(2), Width(7)].
        rope.remove_inclusive(0..0, usize::cmp); // Removes Width(24).
        assert_eq!(rope, [Width(24)].as_slice());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn remove_02() {
        let slice = &[Width(1); 15];
        let mut rope = Rope::from_slice(slice);

        // assert_invariants() below.
        rope.remove_inclusive(3..6, usize::cmp);
        assert_eq!(rope, [Width(1); 12].as_slice());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn remove_03() {
        let mut rope = Rope::from(pseudo_random());

        // Make sure removing an empty range, on a non 0 measure element, does nothing.
        rope.remove_inclusive(45..45, usize::cmp);
        assert_eq!(rope, pseudo_random());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn remove_04() {
        let mut rope = Rope::from(pseudo_random());

        // Make sure removing everything works.
        rope.remove_inclusive(0..135, usize::cmp);
        assert_eq!(rope, [].as_slice());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn remove_05() {
        let mut rope = Rope::from(pseudo_random());

        // Make sure removing a large range works.
        rope.remove_inclusive(3..135, usize::cmp);
        assert_eq!(rope, &pseudo_random()[..2]);

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn remove_06() {
        let mut vec = Vec::from([Width(1); 2]);
        vec.extend_from_slice(&[Width(0); 300]);
        vec.extend_from_slice(&[Width(2); 3]);

        let mut rope = Rope::from(vec);
        rope.remove_inclusive(2..2, usize::cmp);

        assert_eq!(
            rope,
            [Width(1), Width(1), Width(2), Width(2), Width(2)].as_slice()
        );
    }

    #[test]
    #[should_panic]
    fn remove_07() {
        let mut rope = Rope::from(pseudo_random());
        #[allow(clippy::reversed_empty_ranges)]
        rope.remove_inclusive(56..55, usize::cmp); // Wrong ordering of start/end on purpose.
    }

    #[test]
    #[should_panic]
    fn remove_08() {
        let mut rope = Rope::from(pseudo_random());
        rope.remove_inclusive(134..136, usize::cmp); // Removing past the end
    }

    #[test]
    fn split_off_01() {
        let mut rope = Rope::from(pseudo_random());

        let split = rope.split_off(50, usize::cmp);
        assert_eq!(rope, &pseudo_random()[..24]);
        assert_eq!(split, &pseudo_random()[24..]);

        rope.assert_integrity();
        split.assert_integrity();
        rope.assert_invariants();
        split.assert_invariants();
    }

    #[test]
    fn split_off_02() {
        let mut rope = Rope::from(pseudo_random());

        let split = rope.split_off(1, usize::cmp);
        assert_eq!(rope, [Width(1)].as_slice());
        assert_eq!(split, &pseudo_random()[1..]);

        rope.assert_integrity();
        split.assert_integrity();
        rope.assert_invariants();
        split.assert_invariants();
    }

    #[test]
    fn split_off_03() {
        let mut rope = Rope::from(pseudo_random());

        let split = rope.split_off(134, usize::cmp);
        assert_eq!(rope, &pseudo_random()[..69]);
        assert_eq!(split, [Width(0)].as_slice());

        rope.assert_integrity();
        split.assert_integrity();
        rope.assert_invariants();
        split.assert_invariants();
    }

    #[test]
    fn split_off_04() {
        let mut rope = Rope::from(pseudo_random());

        let split = rope.split_off(0, usize::cmp);
        assert_eq!(rope, [].as_slice());
        assert_eq!(split, pseudo_random().as_slice());

        rope.assert_integrity();
        split.assert_integrity();
        rope.assert_invariants();
        split.assert_invariants();
    }

    #[test]
    fn split_off_05() {
        let mut rope = Rope::from(pseudo_random());

        let split = rope.split_off(135, usize::cmp);
        assert_eq!(rope, pseudo_random().as_slice());
        assert_eq!(split, [].as_slice());

        rope.assert_integrity();
        split.assert_integrity();
        rope.assert_invariants();
        split.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn split_off_06() {
        let mut rope = Rope::from(pseudo_random());
        rope.split_off(136, usize::cmp); // One past the end of the rope
    }

    #[test]
    fn append_01() {
        let mut rope = Rope::from_slice(&pseudo_random()[..35]);
        let append = Rope::from_slice(&pseudo_random()[35..]);

        rope.append(append);
        assert_eq!(rope, pseudo_random().as_slice());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn append_02() {
        let mut rope = Rope::from_slice(&pseudo_random()[..68]);
        let append = Rope::from_slice(&[Width(3), Width(0)]);

        rope.append(append);
        assert_eq!(rope, pseudo_random());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn append_03() {
        let mut rope = Rope::from_slice(&[Width(1), Width(2)]);
        let append = Rope::from_slice(&pseudo_random()[2..]);

        rope.append(append);
        assert_eq!(rope, pseudo_random());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn append_04() {
        let mut rope = Rope::from(pseudo_random());
        let append = Rope::from_slice([].as_slice());

        rope.append(append);
        assert_eq!(rope, pseudo_random());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn append_05() {
        let mut rope = Rope::from_slice([].as_slice());
        let append = Rope::from(pseudo_random());

        rope.append(append);
        assert_eq!(rope, pseudo_random());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn measure_to_index_01() {
        let rope = Rope::from(pseudo_random());

        assert_eq!(rope.start_measure_to_index(0, usize::cmp), 0);
        assert_eq!(rope.start_measure_to_index(1, usize::cmp), 1);
        assert_eq!(rope.start_measure_to_index(2, usize::cmp), 1);

        assert_eq!(rope.start_measure_to_index(91, usize::cmp), 47);
        assert_eq!(rope.start_measure_to_index(92, usize::cmp), 47);
        assert_eq!(rope.start_measure_to_index(93, usize::cmp), 48);
        assert_eq!(rope.start_measure_to_index(94, usize::cmp), 49);

        assert_eq!(rope.start_measure_to_index(102, usize::cmp), 51);
        assert_eq!(rope.start_measure_to_index(103, usize::cmp), 51);
    }

    #[test]
    fn from_index_01() {
        let rope = Rope::from(pseudo_random());

        assert_eq!(rope.from_index(0), (0, Width(1)));

        assert_eq!(rope.from_index(67), (132, Width(0)));
        assert_eq!(rope.from_index(68), (132, Width(3)));
        assert_eq!(rope.from_index(69), (135, Width(0)));
    }

    #[test]
    #[should_panic]
    fn from_index_02() {
        let rope = Rope::from(pseudo_random());
        rope.from_index(70);
    }

    #[test]
    #[should_panic]
    fn from_index_03() {
        let rope: Rope<Width> = Rope::from_slice(&[]);
        rope.from_index(0);
    }

    #[test]
    fn from_measure_01() {
        let rope = Rope::from(pseudo_random());

        assert_eq!(rope.from_measure(0, usize::cmp), (0, Width(1)));
        assert_eq!(rope.from_measure(10, usize::cmp), (7, Width(5)));
        assert_eq!(rope.from_measure(18, usize::cmp), (16, Width(8)));
        assert_eq!(rope.from_measure(108, usize::cmp), (108, Width(0)));
    }

    #[test]
    #[should_panic]
    fn from_measure_02() {
        let rope = Rope::from(pseudo_random());
        rope.from_measure(136, usize::cmp);
    }

    #[test]
    #[should_panic]
    fn from_measure_03() {
        let rope: Rope<Width> = Rope::from_slice(&[]);
        rope.from_measure(0, usize::cmp);
    }

    #[test]
    fn chunk_at_index() {
        let rope = Rope::from(pseudo_random());
        let lorem_ipsum = pseudo_random();
        let mut total = lorem_ipsum.as_slice();

        let mut last_chunk = [].as_slice();
        for i in 0..rope.len() {
            let (chunk, index, measure) = rope.chunk_at_index(i);
            assert_eq!(measure, index_to_measure(&lorem_ipsum, index));
            if chunk != last_chunk {
                assert_eq!(chunk, &total[..chunk.len()]);
                total = &total[chunk.len()..];
                last_chunk = chunk;
            }

            let measure_1 = lorem_ipsum.get(i).unwrap();
            let measure_2 = {
                let i2 = i - index;
                chunk.get(i2).unwrap()
            };
            assert_eq!(measure_1, measure_2);
        }
        assert_eq!(total.len(), 0);
    }

    #[test]
    fn chunk_at_measure() {
        let rope = Rope::from(pseudo_random());
        let lorem_ipsum = pseudo_random();
        let mut total = lorem_ipsum.as_slice();

        let mut last_chunk = [].as_slice();
        for i in 0..rope.measure() {
            let (chunk, _, measure) = rope.chunk_at_measure(i, usize::cmp);
            if chunk != last_chunk {
                assert_eq!(chunk, &total[..chunk.len()]);
                total = &total[chunk.len()..];
                last_chunk = chunk;
            }

            let measure_1 = {
                let index_1 = start_measure_to_index(&lorem_ipsum, i, usize::cmp);
                lorem_ipsum.get(index_1).unwrap()
            };
            let measure_2 = {
                let index_2 = start_measure_to_index(chunk, i - measure, usize::cmp);
                chunk.get(index_2)
            };
            if let Some(measure_2) = measure_2 {
                assert_eq!(measure_1, measure_2);
            }
        }
        assert_eq!(total.len(), 0);
    }

    #[test]
    fn measure_slice_01() {
        let rope = Rope::from(pseudo_random());

        let slice = rope.measure_slice(0..rope.measure(), usize::cmp);

        assert_eq!(slice, pseudo_random());
    }

    #[test]
    fn measure_slice_02() {
        let rope = Rope::from(pseudo_random());

        let slice = rope.measure_slice(5..21, usize::cmp);

        assert_eq!(slice, &pseudo_random()[2..10]);
    }

    #[test]
    fn measure_slice_03() {
        let rope = Rope::from(pseudo_random());

        let slice = rope.measure_slice(31..135, usize::cmp);

        assert_eq!(slice, &pseudo_random()[16..70]);
    }

    #[test]
    fn measure_slice_04() {
        let rope = Rope::from(pseudo_random());

        let slice = rope.measure_slice(53..53, usize::cmp);

        assert_eq!([].as_slice(), slice);
    }

    #[test]
    #[should_panic]
    fn measure_slice_05() {
        let rope = Rope::from(pseudo_random());
        #[allow(clippy::reversed_empty_ranges)]
        rope.measure_slice(53..52, usize::cmp); // Wrong ordering on purpose.
    }

    #[test]
    #[should_panic]
    fn measure_slice_06() {
        let rope = Rope::from(pseudo_random());
        rope.measure_slice(134..136, usize::cmp);
    }

    #[test]
    fn index_slice_01() {
        let rope = Rope::from(pseudo_random());

        let slice = rope.index_slice(0..rope.len());

        assert_eq!(pseudo_random(), slice);
    }

    #[test]
    fn index_slice_02() {
        let rope = Rope::from(pseudo_random());

        let slice = rope.index_slice(5..21);

        assert_eq!(&pseudo_random()[5..21], slice);
    }

    #[test]
    fn index_slice_03() {
        let rope = Rope::from(pseudo_random());

        let slice = rope.index_slice(31..55);

        assert_eq!(&pseudo_random()[31..55], slice);
    }

    #[test]
    fn index_slice_04() {
        let rope = Rope::from(pseudo_random());

        let slice = rope.index_slice(53..53);

        assert_eq!([].as_slice(), slice);
    }

    #[test]
    #[should_panic]
    fn index_slice_05() {
        let rope = Rope::from(pseudo_random());
        #[allow(clippy::reversed_empty_ranges)]
        rope.index_slice(53..52); // Wrong ordering on purpose.
    }

    #[test]
    #[should_panic]
    fn index_slice_06() {
        let rope = Rope::from(pseudo_random());
        rope.index_slice(20..72);
    }

    #[test]
    fn eq_rope_01() {
        let rope: Rope<Width> = Rope::from_slice([].as_slice());

        assert_eq!(rope, rope);
    }

    #[test]
    fn eq_rope_02() {
        let rope = Rope::from(pseudo_random());

        assert_eq!(rope, rope);
    }

    #[test]
    fn eq_rope_03() {
        let rope_1 = Rope::from(pseudo_random());
        let mut rope_2 = rope_1.clone();
        rope_2.remove_inclusive(26..27, usize::cmp);
        rope_2.insert(26, Width(1000));

        assert_ne!(rope_1, rope_2);
    }

    #[test]
    fn eq_rope_04() {
        let rope: Rope<Width> = Rope::from_slice([].as_slice());

        assert_eq!(rope, [].as_slice());
        assert_eq!([].as_slice(), rope);
    }

    #[test]
    fn eq_rope_05() {
        let rope = Rope::from(pseudo_random());

        assert_eq!(rope, pseudo_random());
        assert_eq!(pseudo_random(), rope);
    }

    #[test]
    fn eq_rope_06() {
        let mut rope = Rope::from(pseudo_random());
        rope.remove_inclusive(26..27, usize::cmp);
        rope.insert(26, Width(5000));

        assert_ne!(rope, pseudo_random());
        assert_ne!(pseudo_random(), rope);
    }

    #[test]
    fn eq_rope_07() {
        let rope = Rope::from(pseudo_random());
        let slice: Vec<Width> = pseudo_random();

        assert_eq!(rope, slice);
        assert_eq!(slice, rope);
    }

    #[test]
    fn to_vec_01() {
        let rope = Rope::from(pseudo_random());
        let slice: Vec<Width> = (&rope).into();

        assert_eq!(rope, slice);
    }

    #[test]
    fn to_cow_01() {
        use std::borrow::Cow;
        let rope = Rope::from(pseudo_random());
        let cow: Cow<[Width]> = (&rope).into();

        assert_eq!(rope, cow);
    }

    #[test]
    fn to_cow_02() {
        use std::borrow::Cow;
        let rope = Rope::from(pseudo_random());
        let cow: Cow<[Width]> = (rope.clone()).into();

        assert_eq!(rope, cow);
    }

    #[test]
    fn to_cow_03() {
        use std::borrow::Cow;
        let rope = Rope::from_slice(&[Width(1)]);
        let cow: Cow<[Width]> = (&rope).into();

        // Make sure it'slice borrowed.
        if let Cow::Owned(_) = cow {
            panic!("Small Cow conversions should result in a borrow.");
        }

        assert_eq!(rope, cow);
    }

    #[test]
    fn from_rope_slice_01() {
        let rope_1 = Rope::from(pseudo_random());
        let slice = rope_1.measure_slice(.., usize::cmp);
        let rope_2: Rope<Width> = slice.into();

        assert_eq!(rope_1, rope_2);
        assert_eq!(slice, rope_2);
    }

    #[test]
    fn from_rope_slice_02() {
        let rope_1 = Rope::from(pseudo_random());
        let slice = rope_1.measure_slice(0..24, usize::cmp);
        let rope_2: Rope<Width> = slice.into();

        assert_eq!(slice, rope_2);
    }

    #[test]
    fn from_rope_slice_03() {
        let rope_1 = Rope::from(pseudo_random());
        let slice = rope_1.measure_slice(13..89, usize::cmp);
        let rope_2: Rope<Width> = slice.into();

        assert_eq!(slice, rope_2);
    }

    #[test]
    fn from_rope_slice_04() {
        let rope_1 = Rope::from(pseudo_random());
        let slice = rope_1.measure_slice(13..41, usize::cmp);
        let rope_2: Rope<Width> = slice.into();

        assert_eq!(slice, rope_2);
    }

    #[test]
    fn from_iter_01() {
        let rope_1 = Rope::from(pseudo_random());
        let rope_2 = Rope::from_iter(rope_1.chunks());

        assert_eq!(rope_1, rope_2);
    }

    #[test]
    fn is_instance_01() {
        let rope = Rope::from_slice(&[Width(1), Width(2), Width(10), Width(0), Width(0)]);
        let mut c1 = rope.clone();
        let c2 = c1.clone();

        assert!(rope.is_instance(&c1));
        assert!(rope.is_instance(&c2));
        assert!(c1.is_instance(&c2));

        c1.insert_slice(0, &[Width(8)], usize::cmp);

        assert!(!rope.is_instance(&c1));
        assert!(rope.is_instance(&c2));
        assert!(!c1.is_instance(&c2));
    }
}
