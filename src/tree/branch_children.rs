use std::{
    cmp::Ordering,
    fmt,
    iter::{Iterator, Zip},
    slice,
    sync::Arc,
};

use crate::{
    tree::{max_children, max_len, Node, SliceInfo},
    Measurable,
};

/// A fixed-capacity vec of child Arc-pointers and child metadata.
///
/// The unsafe guts of this.is_le() are implemented in BranchChildrenInternal
/// lower down in this file.
#[derive(Clone)]
#[repr(C)]
pub(crate) struct BranchChildren<M>(inner::BranchChildrenInternal<M>)
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized;

impl<M> BranchChildren<M>
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    /// Creates a new empty array.
    pub fn new() -> Self {
        BranchChildren(inner::BranchChildrenInternal::new())
    }

    /// Current length of the array.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether the array is full or not.
    pub fn is_full(&self) -> bool {
        self.len() == max_children::<M, M::Measure>()
    }

    /// Access to the nodes array.
    pub fn nodes(&self) -> &[Arc<Node<M>>] {
        self.0.nodes()
    }

    /// Mutable access to the nodes array.
    pub fn nodes_mut(&mut self) -> &mut [Arc<Node<M>>] {
        self.0.nodes_mut()
    }

    /// Access to the info array.
    pub fn info(&self) -> &[SliceInfo<M::Measure>] {
        self.0.info()
    }

    /// Mutable access to the info array.
    pub fn info_mut(&mut self) -> &mut [SliceInfo<M::Measure>] {
        self.0.info_mut()
    }

    /// Mutable access to both the info and nodes arrays simultaneously.
    pub fn data_mut(&mut self) -> (&mut [SliceInfo<M::Measure>], &mut [Arc<Node<M>>]) {
        self.0.data_mut()
    }

    /// Updates the [`SliceInfo`] of the child at `index`.
    pub fn update_child_info(&mut self, index: usize) {
        let (info, nodes) = self.0.data_mut();
        info[index] = nodes[index].info()
    }

    /// Pushes an item into the end of the array.
    ///
    /// Increases length by one. Panics if already full.
    pub fn push(&mut self, item: (SliceInfo<M::Measure>, Arc<Node<M>>)) {
        self.0.push(item)
    }

    /// Pushes an element onto the end of the array, and then splits it in half,
    /// returning the right half.
    ///
    /// This works even when the array is full.
    pub fn push_split(&mut self, new_child: (SliceInfo<M::Measure>, Arc<Node<M>>)) -> Self {
        let r_count = (self.len() + 1) / 2;
        let l_count = (self.len() + 1) - r_count;

        let mut right = self.split_off(l_count);
        right.push(new_child);
        right
    }

    /// Attempts to merge two nodes, and if it's too much data to merge
    /// equi-distributes it between the two.
    ///
    /// Returns:
    ///
    /// - True: merge was successful.
    /// - False: merge failed, equidistributed instead.
    pub fn merge_distribute(&mut self, index1: usize, index2: usize) -> bool {
        assert!(index1 < index2);
        assert!(index2 < self.len());
        let remove_right = {
            let ((_, node1), (_, node2)) = self.get_two_mut(index1, index2);
            let node1 = Arc::make_mut(node1);
            let node2 = Arc::make_mut(node2);
            match *node1 {
                Node::Leaf(ref mut slice1) => {
                    if let Node::Leaf(ref mut slice2) = *node2 {
                        if (slice1.len() + slice2.len()) <= max_children::<M, M::Measure>() {
                            slice1.push_slice(slice2);
                            true
                        } else {
                            let right = slice1.push_slice_split(slice2);
                            *slice2 = right;
                            false
                        }
                    } else {
                        panic!("Siblings have different node types");
                    }
                }

                Node::Branch(ref mut children1) => {
                    if let Node::Branch(ref mut children2) = *node2 {
                        if (children1.len() + children2.len()) <= max_children::<M, M::Measure>() {
                            for _ in 0..children2.len() {
                                children1.push(children2.remove(0));
                            }
                            true
                        } else {
                            children1.distribute_with(children2);
                            false
                        }
                    } else {
                        panic!("Siblings have different node types");
                    }
                }
            }
        };

        if remove_right {
            self.remove(index2);
            self.update_child_info(index1);
            return true;
        } else {
            self.update_child_info(index1);
            self.update_child_info(index2);
            return false;
        }
    }

    /// Equi-distributes the children between the two child arrays,
    /// preserving ordering.
    pub fn distribute_with(&mut self, other: &mut Self) {
        let r_target_len = (self.len() + other.len()) / 2;
        while other.len() < r_target_len {
            other.insert(0, self.pop());
        }
        while other.len() > r_target_len {
            self.push(other.remove(0));
        }
    }

    /// If the children are leaf nodes, compacts them to take up the fewest
    /// nodes.
    pub fn compact_leaves(&mut self) {
        if !self.nodes()[0].is_leaf() || self.len() < 2 {
            return;
        }

        let mut i = 1;
        while i < self.len() {
            if (self.nodes()[i - 1].leaf_slice().len() + self.nodes()[i].leaf_slice().len())
                <= max_len::<M, M::Measure>()
            {
                // Scope to contain borrows
                {
                    let ((_, node_l), (_, node_r)) = self.get_two_mut(i - 1, i);
                    let slice_l = Arc::make_mut(node_l).leaf_slice_mut();
                    let slice_r = node_r.leaf_slice();
                    slice_l.push_slice(slice_r);
                }
                self.remove(i);
            } else if self.nodes()[i - 1].leaf_slice().len() < max_len::<M, M::Measure>() {
                // Scope to contain borrows
                {
                    let ((_, node_l), (_, node_r)) = self.get_two_mut(i - 1, i);
                    let slice_l = Arc::make_mut(node_l).leaf_slice_mut();
                    let slice_r = Arc::make_mut(node_r).leaf_slice_mut();
                    let split_index_r = max_len::<M, M::Measure>() - slice_l.len();
                    slice_l.push_slice(&slice_r[..split_index_r]);
                    slice_r.truncate_front(split_index_r);
                }
                i += 1;
            } else {
                i += 1;
            }
        }

        for i in 0..self.len() {
            self.update_child_info(i);
        }
    }

    /// Pops an item off the end of the array and returns it.
    ///
    /// Decreases length by one. Panics if already empty.
    pub fn pop(&mut self) -> (SliceInfo<M::Measure>, Arc<Node<M>>) {
        self.0.pop()
    }

    /// Inserts an item into the the array at the given index.
    ///
    /// Increases length by one. Panics if already full. Preserves ordering
    /// of the other items.
    pub fn insert(&mut self, index: usize, item: (SliceInfo<M::Measure>, Arc<Node<M>>)) {
        self.0.insert(index, item)
    }

    /// Inserts an element into a the array, and then splits it in half,
    /// returning the right half.
    ///
    /// This works even when the array is full.
    pub fn insert_split(
        &mut self,
        index: usize,
        item: (SliceInfo<M::Measure>, Arc<Node<M>>),
    ) -> Self {
        assert!(self.len() > 0);
        assert!(index <= self.len());
        let extra = if index < self.len() {
            let extra = self.pop();
            self.insert(index, item);
            extra
        } else {
            item
        };

        self.push_split(extra)
    }

    /// Removes the item at the given index from the the array.
    ///
    /// Decreases length by one. Preserves ordering of the other items.
    pub fn remove(&mut self, index: usize) -> (SliceInfo<M::Measure>, Arc<Node<M>>) {
        self.0.remove(index)
    }

    /// Splits the array in two at `index`, returning the right part of the
    /// split.
    ///
    /// TODO: implement this more efficiently.
    pub fn split_off(&mut self, index: usize) -> Self {
        assert!(index <= self.len());

        let mut other = BranchChildren::new();
        let count = self.len() - index;
        for _ in 0..count {
            other.push(self.remove(index));
        }

        other
    }

    /// Fetches two children simultaneously, returning mutable references
    /// to their info and nodes.
    ///
    /// `index1` must be less than `index2`.
    pub fn get_two_mut(
        &mut self,
        index1: usize,
        index2: usize,
    ) -> (
        (&mut SliceInfo<M::Measure>, &mut Arc<Node<M>>),
        (&mut SliceInfo<M::Measure>, &mut Arc<Node<M>>),
    ) {
        assert!(index1 < index2);
        assert!(index2 < self.len());

        let split_index = index1 + 1;
        let (info, nodes) = self.data_mut();
        let (info1, info2) = info.split_at_mut(split_index);
        let (nodes1, nodes2) = nodes.split_at_mut(split_index);

        (
            (&mut info1[index1], &mut nodes1[index1]),
            (
                &mut info2[index2 - split_index],
                &mut nodes2[index2 - split_index],
            ),
        )
    }

    /// Creates an iterator over the array's items.
    pub fn iter(&self) -> Zip<slice::Iter<SliceInfo<M::Measure>>, slice::Iter<Arc<Node<M>>>> {
        Iterator::zip(self.info().iter(), self.nodes().iter())
    }

    #[allow(clippy::needless_range_loop)]
    pub fn combined_info(&self) -> SliceInfo<M::Measure> {
        let info = self.info();
        let mut acc = SliceInfo::<M::Measure>::new::<M>();

        // Doing this with an explicit loop is notably faster than
        // using an iterator in this case.
        for i in 0..info.len() {
            acc += info[i];
        }

        acc
    }

    /// Returns the child index and left-side-accumulated [`SliceInfo`] of the
    /// first child that matches the given predicate.
    ///
    /// If no child matches the predicate, the last child is returned.
    #[inline(always)]
    pub fn search_by<F>(&self, pred: F) -> (usize, SliceInfo<M::Measure>)
    where
        // (left-accumulated start info, left-accumulated end info)
        F: Fn(SliceInfo<M::Measure>) -> bool,
    {
        debug_assert!(self.len() > 0);

        let mut accum = SliceInfo::<M::Measure>::new::<M>();
        let mut index = 0;
        for info in self.info()[0..(self.len() - 1)].iter() {
            let next_accum = accum + *info;
            if pred(next_accum) {
                break;
            }
            accum = next_accum;
            index += 1;
        }

        (index, accum)
    }

    /// Returns the child index and left-side-accumulated [`SliceInfo`] of the
    /// child that contains the given index.
    ///
    /// One-past-the end is valid, and will return the last child.
    pub fn search_index(&self, index: usize) -> (usize, SliceInfo<M::Measure>) {
        let (index, accum) = self.search_by(|end| index < end.len as usize);

        debug_assert!(
            index <= (accum.len + self.info()[index].len) as usize,
            "Index out of bounds."
        );

        (index, accum)
    }

    /// Returns the child index and left-side-accumulated [`SliceInfo`]o of the
    /// child that contains the given width.
    ///
    /// One-past-the end is valid, and will return the last child.
    pub fn search_start_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> (usize, SliceInfo<M::Measure>) {
        debug_assert!(self.len() > 0);

        let mut accum = SliceInfo::<M::Measure>::new::<M>();
        let mut index = 0;
        for info in self.info()[..self.info().len() - 1].iter() {
            let new_accum = accum + *info;

            if cmp(&measure, &new_accum.measure).is_le() {
                break;
            }

            accum = new_accum;
            index += 1;
        }

        debug_assert!(
            cmp(&measure, &(accum.measure + self.info()[index].measure)).is_le(),
            "Index out of bounds."
        );

        (index, accum)
    }

    /// Returns the child index and left-side-accumulated [`SliceInfo`] of the
    /// child that contains the given width.
    ///
    /// One-past-the end is valid, and will return the last child.
    pub fn search_end_measure(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> (usize, SliceInfo<M::Measure>) {
        // The search uses the `<=` comparison because any slice may end with 0 width
        // elements, and the use of the `<` comparison would leave those behind.
        let (index, accum) = self.search_by(|end| cmp(&measure, &end.measure).is_lt());

        debug_assert!(
            cmp(&measure, &(accum.measure + self.info()[index].measure)).is_le(),
            "Index out of bounds."
        );

        (index, accum)
    }

    /// Same as [`search_start_width()`][Self::search_start_width] above,
    /// except that it only calulates the left-side-accumulated _measure_,
    /// rather than the full [`SliceInfo`].
    ///
    /// Return is (child_index, left_acc_measure)
    ///
    /// One-past-the end is valid, and will return the last child.
    #[inline(always)]
    pub fn search_measure_only(
        &self,
        measure: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> (usize, M::Measure) {
        debug_assert!(self.len() > 0);

        let mut accum = M::Measure::default();
        let mut index = 0;

        for info in self.info()[..self.info().len() - 1].iter() {
            let new_accum = accum + info.measure;

            if cmp(&measure, &new_accum).is_lt() {
                break;
            }
            accum = new_accum;
            index += 1;
        }

        debug_assert!(
            cmp(&measure, &(accum + self.info()[index].measure)).is_le(),
            "Index out of bounds. {:?}, {:?}", measure, accum + self.info()[index].measure
        );

        (index, accum)
    }

    /// Returns the child indices at the start and end of the given measure
    /// range, and returns their left-side-accumulated measure as well.
    ///
    /// Return is:
    /// (
    ///     (left_node_index, left_acc_left_side_width),
    ///     (right_node_index, right_acc_left_side_width),
    /// )
    ///
    /// One-past-the end is valid, and corresponds to the last child.
    #[inline(always)]
    pub fn search_measure_range(
        &self,
        start_bound: M::Measure,
        end_bound: M::Measure,
        cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> ((usize, M::Measure), (usize, M::Measure)) {
        debug_assert!(self.len() > 0);

        let mut accum = M::Measure::default();
        let mut index = 0;

        // Find left child and info
        for info in self.info()[..(self.len() - 1)].iter() {
            let next_accum = accum + info.measure;
            if cmp(&start_bound, &next_accum).is_le() {
                break;
            }
            accum = next_accum;
            index += 1;
        }
        let l_child_i = index;
        let l_acc_info = accum;

        // Find right child and info
        for info in self.info()[index..(self.len() - 1)].iter() {
            let next_accum = accum + info.measure;
            if cmp(&end_bound, &next_accum).is_lt() {
                break;
            }
            accum = next_accum;
            index += 1;
        }

        #[cfg(any(test, debug_assertions))]
        assert!(
            cmp(&end_bound, &(accum + self.info()[index].measure)).is_le(),
            "Index out of bounds."
        );

        ((l_child_i, l_acc_info), (index, accum))
    }

    // Debug function, to help verify tree integrity
    pub fn is_info_accurate(&self) -> bool {
        for (info, node) in self.info().iter().zip(self.nodes().iter()) {
            if *info != node.info() {
                return false;
            }
        }
        true
    }
}

impl<M> fmt::Debug for BranchChildren<M>
where
    M: Measurable + fmt::Debug,
    M::Measure: fmt::Debug,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("NodeChildren")
            .field("len", &self.len())
            .field("info", &&self.info())
            .field("nodes", &&self.nodes())
            .finish()
    }
}

//===========================================================================

/// The unsafe guts of [`BranchChildren<M>`][super::BranchChildren], exposed
/// through a safe API.
///
/// Try to keep this as small as possible, and implement functionality on
/// [`BranchChildren<M>`][super::BranchChildren] via the safe APIs whenever
/// possible.
///
/// It's split out this way because it was too easy to accidentally access the
/// fixed size arrays directly, leading to memory-unsafety bugs when
/// accidentally accessing elements that are semantically out of bounds. This
/// happened once, and it was a pain to track down--as memory safety bugs often
/// are.
mod inner {
    use std::{mem, mem::MaybeUninit, ptr, sync::Arc};

    use super::{max_children, Node, SliceInfo};
    use crate::{tree::max_len, Measurable};

    /// This is essentially a fixed-capacity, stack-allocated [Vec<(M,
    /// SliceInfo)>].
    #[repr(C)]
    pub(crate) struct BranchChildrenInternal<M>
    where
        M: Measurable,
        [(); max_len::<M, M::Measure>()]: Sized,
        [(); max_children::<M, M::Measure>()]: Sized,
    {
        /// An array of the child nodes.
        /// INVARIANT: The nodes from 0..len must be initialized
        nodes: [MaybeUninit<Arc<Node<M>>>; max_children::<M, M::Measure>()],
        /// An array of the child node [`SliceInfo`]s
        /// INVARIANT: The nodes from 0..len must be initialized
        info: [MaybeUninit<SliceInfo<M::Measure>>; max_children::<M, M::Measure>()],
        len: u8,
    }

    impl<M> BranchChildrenInternal<M>
    where
        M: Measurable,
        [(); max_len::<M, M::Measure>()]: Sized,
        [(); max_children::<M, M::Measure>()]: Sized,
    {
        /// Creates a new empty array.
        #[inline(always)]
        pub fn new() -> Self {
            // SAFETY: Uninit data is valid for arrays of MaybeUninit.
            // len is zero, so it's ok for all of them to be uninit
            BranchChildrenInternal {
                nodes: unsafe { MaybeUninit::uninit().assume_init() },
                info: unsafe { MaybeUninit::uninit().assume_init() },
                len: 0,
            }
        }

        /// Current length of the array.
        #[inline(always)]
        pub fn len(&self) -> usize {
            self.len as usize
        }

        /// Access to the nodes array.
        #[inline(always)]
        pub fn nodes(&self) -> &[Arc<Node<M>>] {
            // SAFETY: MaybeUninit<T> is layout compatible with T, and
            // the nodes from 0..len are guaranteed to be initialized
            unsafe { mem::transmute(&self.nodes[..(self.len())]) }
        }

        /// Mutable access to the nodes array.
        #[inline(always)]
        pub fn nodes_mut(&mut self) -> &mut [Arc<Node<M>>] {
            // SAFETY: MaybeUninit<T> is layout compatible with T, and
            // the nodes from 0..len are guaranteed to be initialized
            unsafe { mem::transmute(&mut self.nodes[..(self.len as usize)]) }
        }

        /// Access to the info array.
        #[inline(always)]
        pub fn info(&self) -> &[SliceInfo<M::Measure>] {
            // SAFETY: MaybeUninit<T> is layout compatible with T, and
            // the info from 0..len are guaranteed to be initialized
            unsafe { mem::transmute(&self.info[..(self.len())]) }
        }

        /// Mutable access to the info array.
        #[inline(always)]
        pub fn info_mut(&mut self) -> &mut [SliceInfo<M::Measure>] {
            // SAFETY: MaybeUninit<T> is layout compatible with T, and
            // the info from 0..len are guaranteed to be initialized
            unsafe { mem::transmute(&mut self.info[..(self.len as usize)]) }
        }

        /// Mutable access to both the info and nodes arrays simultaneously.
        #[inline(always)]
        pub fn data_mut(&mut self) -> (&mut [SliceInfo<M::Measure>], &mut [Arc<Node<M>>]) {
            // SAFETY: MaybeUninit<T> is layout compatible with T, and
            // the info from 0..len are guaranteed to be initialized
            (
                unsafe { mem::transmute(&mut self.info[..(self.len as usize)]) },
                unsafe { mem::transmute(&mut self.nodes[..(self.len as usize)]) },
            )
        }

        /// Pushes an item into the end of the array.
        ///
        /// Increases length by one. Panics if already full.
        #[inline(always)]
        pub fn push(&mut self, item: (SliceInfo<M::Measure>, Arc<Node<M>>)) {
            assert!(self.len() < max_children::<M, M::Measure>());
            self.info[self.len()] = MaybeUninit::new(item.0);
            self.nodes[self.len as usize] = MaybeUninit::new(item.1);
            // We have just initialized both info and node and 0..=len, so we can increase
            // it
            self.len += 1;
        }

        /// Pops an item off the end of the array and returns it.
        ///
        /// Decreases length by one. Panics if already empty.
        #[inline(always)]
        pub fn pop(&mut self) -> (SliceInfo<M::Measure>, Arc<Node<M>>) {
            assert!(self.len() > 0);
            self.len -= 1;
            // SAFETY: before this, len was long enough to guarantee that both must be init
            // We just decreased the length, guaranteeing that the elements will never be
            // read again
            (unsafe { self.info[self.len()].assume_init() }, unsafe {
                ptr::read(&self.nodes[self.len()]).assume_init()
            })
        }

        /// Inserts an item into the the array at the given index.
        ///
        /// Increases length by one. Panics if already full. Preserves ordering
        /// of the other items.
        #[inline(always)]
        pub fn insert(&mut self, index: usize, item: (SliceInfo<M::Measure>, Arc<Node<M>>)) {
            assert!(index <= self.len());
            assert!(self.len() < max_children::<M, M::Measure>());

            let len = self.len();
            // This unsafe code simply shifts the elements of the arrays over
            // to make space for the new inserted value. The `.info` array
            // shifting can be done with a safe call to `copy_within()`.
            // However, the `.nodes` array shift cannot, because of the
            // specific drop semantics needed for safety.
            unsafe {
                let ptr = self.nodes.as_mut_ptr();
                ptr::copy(ptr.add(index), ptr.add(index + 1), len - index);
            }
            self.info.copy_within(index..len, index + 1);

            // We have just made space for the two new elements, so insert them
            self.info[index] = MaybeUninit::new(item.0);
            self.nodes[index] = MaybeUninit::new(item.1);
            // Now that all elements from 0..=len are initialized, we can increase the
            // length
            self.len += 1;
        }

        /// Removes the item at the given index from the the array.
        ///
        /// Decreases length by one. Preserves ordering of the other items.
        #[inline(always)]
        pub fn remove(&mut self, index: usize) -> (SliceInfo<M::Measure>, Arc<Node<M>>) {
            assert!(self.len() > 0);
            assert!(index < self.len());

            // Read out the elements, they must not be touched again. We copy the elements
            // after them into them, and decrease the length at the end
            let item = (unsafe { self.info[index].assume_init() }, unsafe {
                ptr::read(&self.nodes[index]).assume_init()
            });

            let len = self.len();
            // This unsafe code simply shifts the elements of the arrays over
            // to fill in the gap left by the removed element. The `.info`
            // array shifting can be done with a safe call to `copy_within()`.
            // However, the `.nodes` array shift cannot, because of the
            // specific drop semantics needed for safety.
            unsafe {
                let ptr = self.nodes.as_mut_ptr();
                ptr::copy(ptr.add(index + 1), ptr.add(index), len - index - 1);
            }
            self.info.copy_within((index + 1)..len, index);

            // Now that the gap is filled, decrease the length
            self.len -= 1;

            return item;
        }
    }

    impl<M> Drop for BranchChildrenInternal<M>
    where
        M: Measurable,
        [(); max_len::<M, M::Measure>()]: Sized,
        [(); max_children::<M, M::Measure>()]: Sized,
    {
        fn drop(&mut self) {
            // The `.nodes` array contains `MaybeUninit` wrappers, which need
            // to be manually dropped if valid. We drop only the valid ones
            // here.
            for node in &mut self.nodes[..self.len as usize] {
                unsafe { ptr::drop_in_place(node.as_mut_ptr()) };
            }
        }
    }

    impl<M> Clone for BranchChildrenInternal<M>
    where
        M: Measurable,
        [(); max_len::<M, M::Measure>()]: Sized,
        [(); max_children::<M, M::Measure>()]: Sized,
    {
        fn clone(&self) -> Self {
            // Create an empty NodeChildrenInternal first, then fill it
            let mut clone_array = BranchChildrenInternal::new();

            // Copy nodes... carefully.
            for (clone_arc, arc) in Iterator::zip(
                clone_array.nodes[..self.len()].iter_mut(),
                self.nodes[..self.len()].iter(),
            ) {
                *clone_arc = MaybeUninit::new(Arc::clone(unsafe { &*arc.as_ptr() }));
            }

            // Copy TextInfo
            for (clone_info, info) in Iterator::zip(
                clone_array.info[..self.len()].iter_mut(),
                self.info[..self.len()].iter(),
            ) {
                *clone_info = *info;
            }

            // Set length
            clone_array.len = self.len;

            // Some sanity checks for debug builds
            #[cfg(debug_assertions)]
            {
                for (a, b) in Iterator::zip(
                    clone_array.info[..clone_array.len()].iter(),
                    self.info[..self.len()].iter(),
                ) {
                    assert_eq!(unsafe { a.assume_init().len }, unsafe {
                        b.assume_init().len
                    },);
                }

                for (a, b) in Iterator::zip(
                    clone_array.nodes[..clone_array.len()].iter(),
                    self.nodes[..clone_array.len()].iter(),
                ) {
                    assert!(Arc::ptr_eq(unsafe { &*a.as_ptr() }, unsafe {
                        &*b.as_ptr()
                    },));
                }
            }

            clone_array
        }
    }
}

//===========================================================================

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        tree::{LeafSlice, Node, SliceInfo},
        Width,
    };

    #[test]
    fn search_width_01() {
        let mut children = BranchChildren::new();
        children.push((
            SliceInfo::<usize>::new::<Width>(),
            Arc::new(Node::Leaf(LeafSlice::from_slice(&[
                Width(1),
                Width(2),
                Width(4),
            ]))),
        ));
        children.push((
            SliceInfo::<usize>::new::<Width>(),
            Arc::new(Node::Leaf(LeafSlice::from_slice(&[Width(0), Width(0)]))),
        ));
        children.push((
            SliceInfo::<usize>::new::<Width>(),
            Arc::new(Node::Leaf(LeafSlice::from_slice(&[Width(9), Width(1)]))),
        ));

        children.update_child_info(0);
        children.update_child_info(1);
        children.update_child_info(2);

        assert_eq!(children.search_start_measure(0, usize::cmp).0, 0);
        assert_eq!(children.search_start_measure(1, usize::cmp).0, 0);
        assert_eq!(children.search_start_measure(0, usize::cmp).1.measure, 0);
        assert_eq!(children.search_start_measure(1, usize::cmp).1.measure, 0);

        assert_eq!(children.search_start_measure(6, usize::cmp).0, 0);
        assert_eq!(children.search_start_measure(7, usize::cmp).0, 0);
        assert_eq!(children.search_start_measure(6, usize::cmp).1.measure, 0);
        assert_eq!(children.search_start_measure(7, usize::cmp).1.measure, 0);

        assert_eq!(children.search_start_measure(7, usize::cmp).0, 0);
        assert_eq!(children.search_start_measure(8, usize::cmp).0, 2);
        assert_eq!(children.search_start_measure(7, usize::cmp).1.measure, 0);
        assert_eq!(children.search_start_measure(8, usize::cmp).1.measure, 7);

        assert_eq!(children.search_start_measure(16, usize::cmp).0, 2);
        assert_eq!(children.search_start_measure(17, usize::cmp).0, 2);
        assert_eq!(children.search_start_measure(16, usize::cmp).1.measure, 7);
        assert_eq!(children.search_start_measure(17, usize::cmp).1.measure, 7);
    }

    #[test]
    #[should_panic]
    fn search_width_02() {
        let mut children = BranchChildren::new();
        children.push((
            SliceInfo::<usize>::new::<Width>(),
            Arc::new(Node::Leaf(LeafSlice::from_slice(&[
                Width(1),
                Width(2),
                Width(4),
            ]))),
        ));
        children.push((
            SliceInfo::<usize>::new::<Width>(),
            Arc::new(Node::Leaf(LeafSlice::from_slice(&[Width(0), Width(0)]))),
        ));
        children.push((
            SliceInfo::<usize>::new::<Width>(),
            Arc::new(Node::Leaf(LeafSlice::from_slice(&[Width(9), Width(1)]))),
        ));

        children.update_child_info(0);
        children.update_child_info(1);
        children.update_child_info(2);

        children.search_start_measure(18, usize::cmp);
    }

    #[test]
    fn search_width_range_01() {
        let mut children = BranchChildren::new();
        children.push((
            SliceInfo::<usize>::new::<Width>(),
            Arc::new(Node::Leaf(LeafSlice::from_slice(&[
                Width(1),
                Width(2),
                Width(4),
            ]))),
        ));
        children.push((
            SliceInfo::<usize>::new::<Width>(),
            Arc::new(Node::Leaf(LeafSlice::from_slice(&[Width(0), Width(0)]))),
        ));
        children.push((
            SliceInfo::<usize>::new::<Width>(),
            Arc::new(Node::Leaf(LeafSlice::from_slice(&[Width(9), Width(1)]))),
        ));

        children.update_child_info(0);
        children.update_child_info(1);
        children.update_child_info(2);

        let at_0_0 = children.search_measure_range(0, 0, usize::cmp);
        let at_7_7 = children.search_measure_range(7, 7, usize::cmp);
        let at_8_8 = children.search_measure_range(8, 8, usize::cmp);
        let at_16_16 = children.search_measure_range(16, 16, usize::cmp);

        assert_eq!((at_0_0.0).0, 0);
        assert_eq!((at_0_0.1).0, 0);
        assert_eq!((at_0_0.0).1, 0);
        assert_eq!((at_0_0.1).1, 0);

        assert_eq!((at_7_7.0).0, 0);
        assert_eq!((at_7_7.1).0, 2);
        assert_eq!((at_7_7.0).1, 0);
        assert_eq!((at_7_7.1).1, 7);

        assert_eq!((at_8_8.0).0, 2);
        assert_eq!((at_8_8.1).0, 2);
        assert_eq!((at_8_8.0).1, 7);
        assert_eq!((at_8_8.1).1, 7);

        assert_eq!((at_16_16.0).0, 2);
        assert_eq!((at_16_16.1).0, 2);
        assert_eq!((at_16_16.0).1, 7);
        assert_eq!((at_16_16.1).1, 7);

        let at_0_7 = children.search_measure_range(0, 7, usize::cmp);
        let at_7_16 = children.search_measure_range(7, 16, usize::cmp);

        assert_eq!((at_0_7.0).0, 0);
        assert_eq!((at_0_7.1).0, 2);
        assert_eq!((at_0_7.0).1, 0);
        assert_eq!((at_0_7.1).1, 7);

        assert_eq!((at_7_16.0).0, 0);
        assert_eq!((at_7_16.1).0, 2);
        assert_eq!((at_7_16.0).1, 0);
        assert_eq!((at_7_16.1).1, 7);

        let at_2_4 = children.search_measure_range(6, 8, usize::cmp);

        assert_eq!((at_2_4.0).0, 0);
        assert_eq!((at_2_4.1).0, 2);
        assert_eq!((at_2_4.0).1, 0);
        assert_eq!((at_2_4.1).1, 7);
    }

    #[test]
    #[should_panic]
    fn search_index_range_02() {
        let mut children = BranchChildren::new();
        children.push((
            SliceInfo::<usize>::new::<Width>(),
            Arc::new(Node::Leaf(LeafSlice::from_slice(&[
                Width(1),
                Width(2),
                Width(4),
            ]))),
        ));
        children.push((
            SliceInfo::<usize>::new::<Width>(),
            Arc::new(Node::Leaf(LeafSlice::from_slice(&[Width(0), Width(0)]))),
        ));
        children.push((
            SliceInfo::<usize>::new::<Width>(),
            Arc::new(Node::Leaf(LeafSlice::from_slice(&[Width(9), Width(1)]))),
        ));

        children.update_child_info(0);
        children.update_child_info(1);
        children.update_child_info(2);

        children.search_measure_range(17, 18, usize::cmp);
    }
}
