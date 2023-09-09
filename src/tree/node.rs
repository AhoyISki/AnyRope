use std::{cmp::Ordering, sync::Arc};

use crate::{
    fallible_min,
    slice_utils::{end_measure_to_index, index_to_measure, start_measure_to_index},
    tree::{
        max_children, max_len, min_children, min_len, BranchChildren, Count, LeafSlice, SliceInfo,
    },
    Measurable,
};

#[derive(Debug, Clone)]
#[repr(u8, C)]
pub(crate) enum Node<M>
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    Leaf(LeafSlice<M>),
    Branch(BranchChildren<M>),
}

impl<M> Node<M>
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    /// Creates an empty [`Node<M>`].
    #[inline(always)]
    pub fn new() -> Self {
        Node::Leaf(LeafSlice::from_slice(&[]))
    }

    /// Total number of items in the [`Node<M>`].
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.info().len as usize
    }

    /// Total [`M::Measure`] in the [`Node<M>`]
    ///
    /// [`M::Measure`]: Measurable::Measure
    #[inline(always)]
    pub fn measure(&self) -> M::Measure {
        self.info().measure
    }

    /// Fetches a chunk mutably, and allows it to be edited via a closure.
    ///
    /// There are three parameters:
    /// - width: the chunk that contains this width is fetched,
    /// - node_info: this is the [`SliceInfo`] of the node it's being called on.
    ///   This makes it a little awkward to call, but is needed since it's
    ///   actually the parent node that contains the [`SliceInfo`], so the info
    ///   needs to be passed in.
    /// - edit: the closure that receives the chunk and does the edits.
    ///
    /// The closure is effectively the termination case for the recursion,
    /// and takes essentially same parameters and returns the same things as
    /// the method itself. In particular, the closure receives the width offset
    /// of the width within the given chunk and the [`SliceInfo`] of the chunk.
    /// The main difference is that it receives a [`LeafSlice<M>`] instead of a
    /// node.
    ///
    /// The closure is expected to return the updated [`SliceInfo`] of the
    /// [`Node<M>`], and if the node had to be split, then it also returns
    /// the right-hand [`Node<M>`] along with its [`SliceInfo`] as well.
    ///
    /// The main method call will then return the total updated [`SliceInfo`]
    /// for the whole tree, and a new [`Node<M>`] only if the whole tree had
    /// to be split. It is up to the caller to check for that new
    /// [`Node<M>`], and handle it by creating a new root with both the
    /// original [`Node<M>`] and the new node as children.
    pub fn edit_chunk_at_measure<F>(
        &mut self,
        measure: M::Measure,
        cmp: &impl Fn(&M::Measure, &M::Measure) -> Ordering,
        info: SliceInfo<M::Measure>,
        mut edit: F,
    ) -> (
        SliceInfo<M::Measure>,
        Option<(SliceInfo<M::Measure>, Arc<Node<M>>)>,
    )
    where
        F: FnMut(
            M::Measure,
            SliceInfo<M::Measure>,
            &mut LeafSlice<M>,
        ) -> (
            SliceInfo<M::Measure>,
            Option<(SliceInfo<M::Measure>, Arc<Node<M>>)>,
        ),
    {
        match self {
            Node::Leaf(slice) => edit(measure, info, slice),
            Node::Branch(children) => {
                // Compact leaf children if we're very close to maximum leaf
                // fragmentation. This basically guards against excessive memory
                // ballooning when repeatedly appending to the end of a rope.
                // The constant here was arrived at experimentally, and is otherwise
                // fairly arbitrary.
                const fn frag_min_bytes<M: Measurable>() -> usize {
                    (max_len::<M, M::Measure>() * min_children::<M, M::Measure>())
                        + (max_len::<M, M::Measure>() / 32)
                }
                if children.is_full()
                    && children.nodes()[0].is_leaf()
                    && (children.combined_info().len as usize) < frag_min_bytes::<M>()
                {
                    children.compact_leaves();
                }

                // Find the child we care about.
                let (child_i, acc_measure) = children.search_measure_only(measure, cmp);
                let child_info = children.info()[child_i];

                // Recurse into the child.
                let (l_info, residual) = Arc::make_mut(&mut children.nodes_mut()[child_i])
                    .edit_chunk_at_measure(measure - acc_measure, cmp, child_info, edit);

                children.info_mut()[child_i] = l_info;

                // Handle the residual node if there is one and return.
                if let Some((r_info, r_node)) = residual {
                    if children.len() < max_children::<M, M::Measure>() {
                        children.insert(child_i + 1, (r_info, r_node));
                        (info - child_info + l_info + r_info, None)
                    } else {
                        let r = children.insert_split(child_i + 1, (r_info, r_node));
                        let r_info = r.combined_info();
                        (
                            children.combined_info(),
                            Some((r_info, Arc::new(Node::Branch(r)))),
                        )
                    }
                } else {
                    (info - child_info + l_info, None)
                }
            }
        }
    }

    /// Removes elements in the range `start_index..end_index`.
    ///
    /// The parameters `left_edge` and `right_edge`, if true, remove 0 width
    /// elements at the edges of the range.
    ///
    /// Returns (in this order):
    /// - The updated [`SliceInfo`] for the node.
    /// - Whether [`fix_tree_seam()`][Node::fix_tree_seam] needs to be run after
    ///   this.
    ///
    /// WARNING: does not correctly handle all slice being removed. That
    /// should be special-cased in calling code.
    pub fn remove_range(
        &mut self,
        start: M::Measure,
        end: M::Measure,
        cmp: &impl Fn(&M::Measure, &M::Measure) -> Ordering,
        incl_left: bool,
        incl_right: bool,
    ) -> (SliceInfo<M::Measure>, bool) {
        let info = self.info();
        match self {
            Node::Leaf(slice) => remove_from_slice(slice, start, end, cmp, incl_left, incl_right),
            Node::Branch(children) => {
                remove_from_children(children, start, end, cmp, incl_left, incl_right, info)
            }
        }
    }

    pub fn append_at_depth(&mut self, mut other: Arc<Node<M>>, depth: usize) -> Option<Arc<Self>> {
        if depth == 0 {
            if let Node::Branch(ref mut children_l) = *self {
                if let Node::Branch(ref mut children_r) = *Arc::make_mut(&mut other) {
                    if (children_l.len() + children_r.len()) <= max_children::<M, M::Measure>() {
                        for _ in 0..children_r.len() {
                            children_l.push(children_r.remove(0));
                        }
                        return None;
                    } else {
                        children_l.distribute_with(children_r);
                        // Return lower down, to avoid borrow-checker.
                    }
                } else {
                    panic!("Tree-append siblings have differing types.");
                }
            } else if !other.is_leaf() {
                panic!("Tree-append siblings have differing types.");
            }

            return Some(other);
        } else if let Node::Branch(ref mut children) = *self {
            let last_i = children.len() - 1;
            let residual =
                Arc::make_mut(&mut children.nodes_mut()[last_i]).append_at_depth(other, depth - 1);
            children.update_child_info(last_i);
            if let Some(extra_node) = residual {
                if children.len() < max_children::<M, M::Measure>() {
                    children.push((extra_node.info(), extra_node));
                    return None;
                } else {
                    let r_children = children.push_split((extra_node.info(), extra_node));
                    return Some(Arc::new(Node::Branch(r_children)));
                }
            } else {
                return None;
            }
        } else {
            panic!("Reached leaf before getting to target depth.");
        }
    }

    pub fn prepend_at_depth(&mut self, other: Arc<Node<M>>, depth: usize) -> Option<Arc<Self>> {
        if depth == 0 {
            match *self {
                Node::Leaf(_) => {
                    if !other.is_leaf() {
                        panic!("Tree-append siblings have differing types.");
                    } else {
                        return Some(other);
                    }
                }
                Node::Branch(ref mut children_r) => {
                    let mut other = other;
                    if let Node::Branch(ref mut children_l) = *Arc::make_mut(&mut other) {
                        if (children_l.len() + children_r.len()) <= max_children::<M, M::Measure>()
                        {
                            for _ in 0..children_l.len() {
                                children_r.insert(0, children_l.pop());
                            }
                            return None;
                        } else {
                            children_l.distribute_with(children_r);
                            // Return lower down, to avoid borrow-checker.
                        }
                    } else {
                        panic!("Tree-append siblings have differing types.");
                    }
                    return Some(other);
                }
            }
        } else if let Node::Branch(ref mut children) = *self {
            let residual =
                Arc::make_mut(&mut children.nodes_mut()[0]).prepend_at_depth(other, depth - 1);
            children.update_child_info(0);
            if let Some(extra_node) = residual {
                if children.len() < max_children::<M, M::Measure>() {
                    children.insert(0, (extra_node.info(), extra_node));
                    return None;
                } else {
                    let mut r_children = children.insert_split(0, (extra_node.info(), extra_node));
                    std::mem::swap(children, &mut r_children);
                    return Some(Arc::new(Node::Branch(r_children)));
                }
            } else {
                return None;
            }
        } else {
            panic!("Reached leaf before getting to target depth.");
        }
    }

    /// Splits the [`Node<M>`] at `measure`, returning the right side of the
    /// split.
    pub fn end_split(
        &mut self,
        measure: M::Measure,
        cmp: &impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Self {
        debug_assert!(measure != M::Measure::default());
        debug_assert!(measure != (self.info().measure));
        match *self {
            Node::Leaf(ref mut slice) => {
                let index = end_measure_to_index(slice, measure, cmp);
                Node::Leaf(slice.split_off(index))
            }
            Node::Branch(ref mut children) => {
                let (child_i, acc_info) = children.search_end_measure(measure, cmp);
                let child_info = children.info()[child_i];

                if measure == acc_info.measure {
                    Node::Branch(children.split_off(child_i))
                } else if measure == (acc_info.measure + child_info.measure) {
                    Node::Branch(children.split_off(child_i + 1))
                } else {
                    let mut r_children = children.split_off(child_i + 1);

                    // Recurse
                    let r_node = Arc::make_mut(&mut children.nodes_mut()[child_i])
                        .end_split(measure - acc_info.measure, cmp);

                    r_children.insert(0, (r_node.info(), Arc::new(r_node)));

                    children.update_child_info(child_i);
                    r_children.update_child_info(0);

                    Node::Branch(r_children)
                }
            }
        }
    }

    /// Splits the [`Node<M>`] index `width`, returning the right side of the
    /// split.
    pub fn start_split(
        &mut self,
        measure: M::Measure,
        cmp: &impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> Self {
        debug_assert!(measure != M::Measure::default());
        debug_assert!(measure != (self.info().measure));
        match *self {
            Node::Leaf(ref mut slice) => {
                let index = start_measure_to_index(slice, measure, cmp);
                Node::Leaf(slice.split_off(index))
            }
            Node::Branch(ref mut children) => {
                let (child_i, acc_info) = children.search_start_measure(measure, cmp);
                let child_info = children.info()[child_i];

                if measure == acc_info.measure {
                    Node::Branch(children.split_off(child_i))
                } else if measure == (acc_info.measure + child_info.measure) {
                    Node::Branch(children.split_off(child_i + 1))
                } else {
                    let mut r_children = children.split_off(child_i + 1);

                    // Recurse
                    let r_node = Arc::make_mut(&mut children.nodes_mut()[child_i])
                        .end_split(measure - acc_info.measure, cmp);

                    r_children.insert(0, (r_node.info(), Arc::new(r_node)));

                    children.update_child_info(child_i);
                    r_children.update_child_info(0);

                    Node::Branch(r_children)
                }
            }
        }
    }

    /// Returns the chunk that contains the given index, and the [`SliceInfo`]
    /// corresponding to the start of the chunk.
    pub fn get_chunk_at_index(&self, mut index: usize) -> (&[M], SliceInfo<M::Measure>) {
        let mut node = self;
        let mut info = SliceInfo::<M::Measure>::new::<M>();

        loop {
            match *node {
                Node::Leaf(ref slice) => {
                    return (slice, info);
                }
                Node::Branch(ref children) => {
                    let (child_i, acc_info) = children.search_index(index);
                    info += acc_info;
                    node = &*children.nodes()[child_i];
                    index -= acc_info.len as usize;
                }
            }
        }
    }

    /// Returns the chunk that contains the given width, and the [`SliceInfo`]
    /// corresponding to the start of the chunk.
    pub fn get_first_chunk_at_measure(
        &self,
        mut measure: M::Measure,
        cmp: &impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> (&[M], SliceInfo<M::Measure>) {
        let mut node = self;
        let mut info = SliceInfo::<M::Measure>::new::<M>();

        loop {
            match *node {
                Node::Leaf(ref slice) => {
                    return (slice, info);
                }
                Node::Branch(ref children) => {
                    let (child_i, acc_info) = children.search_start_measure(measure, cmp);
                    info += acc_info;

                    node = &*children.nodes()[child_i];
                    measure = measure - acc_info.measure;
                }
            }
        }
    }

    /// Returns the chunk that contains the given width, and the [`SliceInfo`]
    /// corresponding to the start of the chunk.
    pub fn get_last_chunk_at_measure(
        &self,
        mut measure: M::Measure,
        cmp: &impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> (&[M], SliceInfo<M::Measure>) {
        let mut node = self;
        let mut info = SliceInfo::<M::Measure>::new::<M>();

        loop {
            match *node {
                Node::Leaf(ref slice) => {
                    return (slice, info);
                }
                Node::Branch(ref children) => {
                    let (child_i, acc_info) = children.search_end_measure(measure, cmp);
                    info += acc_info;

                    node = &*children.nodes()[child_i];
                    measure = measure - acc_info.measure;
                }
            }
        }
    }

    /// Returns the [`SliceInfo`] at the given starting width sum.
    #[inline(always)]
    pub fn start_measure_to_slice_info(
        &self,
        measure: M::Measure,
        cmp: &impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> SliceInfo<M::Measure> {
        let (chunk, info) = self.get_first_chunk_at_measure(measure, cmp);
        let bi = start_measure_to_index(chunk, measure - info.measure, cmp);
        SliceInfo {
            len: info.len + bi as Count,
            measure,
        }
    }

    /// Returns the [`SliceInfo`] at the given ending width sum.
    #[inline(always)]
    pub fn end_measure_to_slice_info(
        &self,
        measure: M::Measure,
        cmp: &impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> SliceInfo<M::Measure> {
        let (chunk, info) = self.get_last_chunk_at_measure(measure, cmp);
        let bi = end_measure_to_index(chunk, measure - info.measure, cmp);
        SliceInfo {
            len: info.len + bi as Count,
            measure,
        }
    }

    /// Returns the [`SliceInfo`] at the given index.
    #[inline(always)]
    pub fn index_to_slice_info(&self, index: usize) -> SliceInfo<M::Measure> {
        let (chunk, info) = self.get_chunk_at_index(index);
        let measure = index_to_measure(chunk, index - info.len as usize);
        SliceInfo {
            len: index as Count,
            measure: info.measure + measure,
        }
    }

    pub fn info(&self) -> SliceInfo<M::Measure> {
        match *self {
            Node::Leaf(ref slice) => SliceInfo::<M::Measure>::from_slice(slice),
            Node::Branch(ref children) => children.combined_info(),
        }
    }

    //-----------------------------------------

    pub fn child_count(&self) -> usize {
        if let Node::Branch(ref children) = *self {
            children.len()
        } else {
            panic!()
        }
    }

    pub fn children(&self) -> &BranchChildren<M> {
        match *self {
            Node::Branch(ref children) => children,
            _ => panic!(),
        }
    }

    pub fn children_mut(&mut self) -> &mut BranchChildren<M> {
        match *self {
            Node::Branch(ref mut children) => children,
            _ => panic!(),
        }
    }

    pub fn leaf_slice(&self) -> &[M] {
        match *self {
            Node::Leaf(ref slice) => slice,
            _ => unreachable!(),
        }
    }

    pub fn leaf_slice_mut(&mut self) -> &mut LeafSlice<M> {
        match *self {
            Node::Leaf(ref mut slice) => slice,
            _ => panic!(),
        }
    }

    pub fn is_leaf(&self) -> bool {
        match *self {
            Node::Leaf(_) => true,
            Node::Branch(_) => false,
        }
    }

    pub fn is_undersized(&self) -> bool {
        match *self {
            Node::Leaf(ref slice) => slice.len() < min_len::<M, M::Measure>(),
            Node::Branch(ref children) => children.len() < min_children::<M, M::Measure>(),
        }
    }

    /// How many nodes deep the tree is.
    ///
    /// This counts root and leafs. For example, a single leaf node
    /// has depth 1.
    pub fn depth(&self) -> usize {
        let mut node = self;
        let mut depth = 0;

        loop {
            match *node {
                Node::Leaf(_) => return depth,
                Node::Branch(ref children) => {
                    depth += 1;
                    node = &*children.nodes()[0];
                }
            }
        }
    }

    /// Debugging tool to make sure that all of the meta-data of the
    /// tree is consistent with the actual data.
    pub fn assert_integrity(&self) {
        match *self {
            Node::Leaf(_) => {}
            Node::Branch(ref children) => {
                for (info, node) in children.iter() {
                    assert_eq!(info.len, node.info().len);
                    node.assert_integrity();
                }
            }
        }
    }

    /// Checks that the entire tree is the same height everywhere.
    pub fn assert_balance(&self) -> usize {
        // Depth, child count, and leaf node emptiness
        match *self {
            Node::Leaf(_) => 1,
            Node::Branch(ref children) => {
                let first_depth = children.nodes()[0].assert_balance();
                for node in &children.nodes()[1..] {
                    assert_eq!(node.assert_balance(), first_depth);
                }
                first_depth + 1
            }
        }
    }

    /// Checks that all internal nodes have the minimum number of
    /// children and all non-root leaf nodes are non-empty.
    pub fn assert_node_size(&self, is_root: bool) {
        match *self {
            Node::Leaf(ref slice) => {
                // Leaf size
                if !is_root {
                    assert!(slice.len() > 0);
                }
            }
            Node::Branch(ref children) => {
                // Child count
                if is_root {
                    assert!(children.len() > 1);
                } else {
                    assert!(children.len() >= min_children::<M, M::Measure>());
                }

                for node in children.nodes() {
                    node.assert_node_size(false);
                }
            }
        }
    }

    /// Fixes dangling nodes down the left side of the tree.
    ///
    /// Returns whether it did anything or not that would affect the
    /// parent.
    pub fn zip_fix_left(&mut self) -> bool {
        if let Node::Branch(ref mut children) = *self {
            let mut did_stuff = false;
            loop {
                let do_merge = (children.len() > 1)
                    && match *children.nodes()[0] {
                        Node::Leaf(ref slice) => slice.len() < min_len::<M, M::Measure>(),
                        Node::Branch(ref children2) => {
                            children2.len() < min_children::<M, M::Measure>()
                        }
                    };

                if do_merge {
                    did_stuff |= children.merge_distribute(0, 1);
                }

                if !Arc::make_mut(&mut children.nodes_mut()[0]).zip_fix_left() {
                    break;
                }
            }
            did_stuff
        } else {
            false
        }
    }

    /// Fixes dangling nodes down the right side of the tree.
    ///
    /// Returns whether it did anything or not that would affect the
    /// parent. True: did stuff, false: didn't do stuff
    pub fn zip_fix_right(&mut self) -> bool {
        if let Node::Branch(ref mut children) = *self {
            let mut did_stuff = false;
            loop {
                let last_i = children.len() - 1;
                let do_merge = (children.len() > 1)
                    && match *children.nodes()[last_i] {
                        Node::Leaf(ref slice) => slice.len() < min_len::<M, M::Measure>(),
                        Node::Branch(ref children2) => {
                            children2.len() < min_children::<M, M::Measure>()
                        }
                    };

                if do_merge {
                    did_stuff |= children.merge_distribute(last_i - 1, last_i);
                }

                if !Arc::make_mut(children.nodes_mut().last_mut().unwrap()).zip_fix_right() {
                    break;
                }
            }
            did_stuff
        } else {
            false
        }
    }

    /// Fixes up the tree after [`remove_range()`][Node::remove_range] or
    /// [`Rope::append()`].
    /// Takes the width of the start of the removal range.
    ///
    /// Returns whether it did anything or not that would affect the
    /// parent. True: did stuff, false: didn't do stuff
    pub fn fix_tree_seam(
        &mut self,
        measure: M::Measure,
        cmp: &impl Fn(&M::Measure, &M::Measure) -> Ordering,
    ) -> bool {
        if let Node::Branch(ref mut children) = *self {
            let mut did_stuff = false;
            loop {
                // Do merging
                if children.len() > 1 {
                    let (child_i, start_info) = children.search_start_measure(measure, cmp);
                    let mut do_merge = match *children.nodes()[child_i] {
                        Node::Leaf(ref slice) => slice.len() < min_len::<M, M::Measure>(),
                        Node::Branch(ref children2) => {
                            children2.len() < min_children::<M, M::Measure>()
                        }
                    };

                    if child_i == 0 {
                        if do_merge {
                            did_stuff |= children.merge_distribute(0, 1);
                        }
                    } else {
                        do_merge |= {
                            cmp(&start_info.measure, &measure).is_eq()
                                && match *children.nodes()[child_i - 1] {
                                    Node::Leaf(ref slice) => {
                                        slice.len() < min_len::<M, M::Measure>()
                                    }
                                    Node::Branch(ref children2) => {
                                        children2.len() < min_children::<M, M::Measure>()
                                    }
                                }
                        };
                        if do_merge {
                            let res = children.merge_distribute(child_i - 1, child_i);
                            did_stuff |= res
                        }
                    }
                }

                // Do recursion
                let (child_i, start_info) = children.search_start_measure(measure, cmp);

                if cmp(&start_info.measure, &measure).is_eq() && child_i != 0 {
                    let tmp = children.info()[child_i - 1].measure;
                    let effect_1 = Arc::make_mut(&mut children.nodes_mut()[child_i - 1])
                        .fix_tree_seam(tmp, cmp);
                    let effect_2 = Arc::make_mut(&mut children.nodes_mut()[child_i])
                        .fix_tree_seam(M::Measure::default(), cmp);
                    if (!effect_1) && (!effect_2) {
                        break;
                    }
                } else if !Arc::make_mut(&mut children.nodes_mut()[child_i])
                    .fix_tree_seam(measure - start_info.measure, cmp)
                {
                    break;
                }
            }
            debug_assert!(children.is_info_accurate());
            did_stuff
        } else {
            false
        }
    }
}

fn remove_from_slice<M>(
    slice: &mut LeafSlice<M>,
    start: M::Measure,
    end: M::Measure,
    cmp: &impl Fn(&M::Measure, &M::Measure) -> Ordering,
    incl_left: bool,
    incl_right: bool,
) -> (SliceInfo<M::Measure>, bool)
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    let start_index = if incl_left {
        start_measure_to_index(slice, start, cmp)
    } else {
        end_measure_to_index(slice, start, cmp)
    };
    // In this circumstance, nothing needs to be done, since we're removing
    // in the middle of an element.
    let non_zero_width = slice
        .get(start_index)
        .map(|m| cmp(&m.measure(), &M::Measure::default()).is_gt())
        .unwrap_or(true);

    if cmp(&start, &end).is_eq() && non_zero_width {
        return (SliceInfo::<M::Measure>::from_slice(slice), false);
    }

    let end_index = if incl_right {
        end_measure_to_index(slice, end, cmp)
    } else {
        start_measure_to_index(slice, end, cmp)
    };

    slice.remove_range(start_index, end_index);
    (SliceInfo::<M::Measure>::from_slice(slice), false)
}

fn remove_from_children<M>(
    children: &mut BranchChildren<M>,
    start: M::Measure,
    end: M::Measure,
    cmp: &impl Fn(&M::Measure, &M::Measure) -> Ordering,
    incl_left: bool,
    incl_right: bool,
    info: SliceInfo<M::Measure>,
) -> (SliceInfo<M::Measure>, bool)
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    let ((left_child, left_accum), (right_child, right_accum)) =
        children.search_measure_range(start, end, cmp);

    if left_child == right_child {
        let child_info = children.info()[left_child];
        let (new_info, mut needs_fix) = handle_measure_range(
            children, left_child, left_accum, start, end, cmp, incl_left, incl_right,
        );

        if children.len() > 0 {
            merge_child(children, left_child);

            // If we couldn't get all children >= minimum size, then
            // we'll need to fix that later.
            if children.nodes()[left_child.min(children.len() - 1)].is_undersized() {
                needs_fix = true;
            }
        }

        (info - child_info + new_info, needs_fix)
    // We're dealing with more than one child.
    } else {
        let mut needs_fix = false;

        // Calculate the start..end range of nodes to be removed.
        let first = left_child + 1;
        let (last, right_child_exists) = {
            let right_measure = children.info()[right_child].measure;
            if right_accum + right_measure == end {
                (right_child + 1, false)
            } else {
                (right_child, true)
            }
        };

        // Remove the children
        for _ in first..last {
            children.remove(first);
        }

        let between_two_children =
            cmp(&start, &right_accum).is_lt() && cmp(&right_accum, &end).is_lt();

        // Handle right child
        if right_child_exists {
            let (_, fix) = handle_measure_range(
                children,
                first,
                right_accum,
                start,
                end,
                cmp,
                incl_left || between_two_children,
                incl_right,
            );
            needs_fix |= fix;
        }

        // Handle left child
        let (_, fix) = {
            handle_measure_range(
                children,
                left_child,
                left_accum,
                start,
                end,
                cmp,
                incl_left,
                incl_right || between_two_children,
            )
        };
        needs_fix |= fix;

        if children.len() > 0 {
            // Handle merging
            let merge_extent = 1 + if right_child_exists { 1 } else { 0 };
            for i in (left_child..(left_child + merge_extent)).rev() {
                merge_child(children, i);
            }

            // If we couldn't get all children >= minimum size, then
            // we'll need to fix that later.
            if children.nodes()[left_child.min(children.len() - 1)].is_undersized() {
                needs_fix = true;
            }
        }

        (children.combined_info(), needs_fix)
    }
}

/// Shared code for handling children.
/// Returns (in this order):
///
/// - Whether the tree may need invariant fixing.
/// - Updated SliceInfo of the node.
fn handle_measure_range<M>(
    children: &mut BranchChildren<M>,
    child_i: usize,
    accum: M::Measure,
    start_measure: M::Measure,
    end_measure: M::Measure,
    cmp: &impl Fn(&M::Measure, &M::Measure) -> Ordering,
    incl_left: bool,
    incl_right: bool,
) -> (SliceInfo<M::Measure>, bool)
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    // Recurse into child
    let child_measure = children.info()[child_i].measure;
    let (new_info, needs_fix) = Arc::make_mut(&mut children.nodes_mut()[child_i]).remove_range(
        start_measure - fallible_min(accum, start_measure),
        fallible_min(end_measure - accum, child_measure),
        cmp,
        incl_left,
        incl_right,
    );

    // Handle result
    if new_info.len == 0 {
        children.remove(child_i);
    } else {
        children.info_mut()[child_i] = new_info;
    }

    (new_info, needs_fix)
}

/// Merges a child with its sibling.
fn merge_child<M>(children: &mut BranchChildren<M>, child_i: usize)
where
    M: Measurable,
    [(); max_len::<M, M::Measure>()]: Sized,
    [(); max_children::<M, M::Measure>()]: Sized,
{
    if child_i < children.len() && children.len() > 1 && children.nodes()[child_i].is_undersized() {
        if child_i == 0 {
            children.merge_distribute(child_i, child_i + 1);
        } else {
            children.merge_distribute(child_i - 1, child_i);
        }
    }
}
