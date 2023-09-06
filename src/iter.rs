//! Iterators over a [`Rope<M>`]'s data.
//!
//! The iterators in Any-Ropey can be created from both [`Rope<M>`]s and
//! [`RopeSlice<M>`]s. When created from a [`RopeSlice<M>`], they iterate over
//! only the data that the [`RopeSlice<M>`] refers to.  For the [`Chunks`]
//! iterator, the data of the first and last yielded item will be correctly
//! truncated to match the bounds of the [`RopeSlice<M>`].
//!
//! # Reverse iteration
//!
//! All iterators in Ropey operate as a cursor that can move both forwards
//! and backwards over its contents.  This can be accomplished via the
//! `next()` and `prev()` methods on each iterator, or by using the `reverse()`
//! or `reversed()` methods to change the iterator's direction.
//!
//! Conceptually, an iterator in Any-Ropey is always positioned *between* the
//! elements it iterates over, and returns an element when it jumps over it
//! via the `next()` or `prev()` methods.
//!
//! For example, given the slice `[Width(1), Width(2), Width(5)]` and a an
//! iterator starting at the beginning of the slice, you would get the following
//! sequence of states and return values by repeatedly calling `next()` (the
//! vertical bar represents the position of the iterator):
//!
//! 0. `[|Width(1), Width(2), Width(5)]`
//! 1. `[Width(1), |Width(2), Width(5)] -> Some(Width(1))`
//! 2. `[Width(1), Width(2), |Width(5)] -> Some(Width(2))`
//! 3. `[Width(1), Width(2), Width(5)|] -> Some(Width(5))`
//! 4. `[Width(1), Width(2), Width(5)|] -> None`
//!
//! The `prev()` method operates identically, except moving in the opposite
//! direction.  And `reverse()` simply swaps the behavior of `prev()` and
//! `next()`.
//!
//! # Creating iterators at any position
//!
//! Iterators in Ropey can be created starting at any position in the rope.
//! This is accomplished with the [`Iter<M>`] and [`Chunks<M>`] iterators, which
//! can be created by various functions on a [`Rope<M>`].
//!
//! When an iterator is created this way, it is positioned such that a call to
//! `next()` will return the specified element, and a call to `prev()` will
//! return the element just before the specified one.
//!
//! Importantly, iterators created this way still have access to the entire
//! contents of the [`Rope<M>`]/[`RopeSlice<M>`] they were created from and the
//! contents before the specified position is not truncated.  For example, you
//! can create an [`Iter<M>`] iterator starting at the end of a [`Rope<M>`], and
//! then use the [`prev()`][Iter::prev] method to iterate backwards over all of
//! that [`Rope<M>`]'s elements.
//!
//! # A possible point of confusion
//!
//! The Rust standard library has an iterator trait [`DoubleEndedIterator`] with
//! a method [`rev()`]. While this method's name is //! very similar to Ropey's
//! [`reverse()`][Iter::reverse] method, its behavior is very different.
//!
//! [`DoubleEndedIterator`] actually provides two iterators: one starting at
//! each end of the collection, moving in opposite directions towards each
//! other. Calling [`rev()`] switches between those two iterators, changing not
//! only the direction of iteration but also its current position in the
//! collection.
//!
//! The [`reverse()`][Iter::reverse] method on AnyRopey's iterators, on the
//! other hand, reverses the direction of the iterator in-place, without
//! changing its position in the rope.
//!
//! [`Rope<M>`]: crate::rope::Rope
//! [`RopeSlice<M>`]: crate::slice::RopeSlice
//! [`rev()`]: Iterator::rev

use std::sync::Arc;

use crate::{
    Measurable,
    slice_utils::{index_to_width, start_width_to_index, width_of},
    tree::{max_children, max_len, Node, SliceInfo},
};

//==========================================================

/// An iterator over a [`Rope<M>`][crate::rope::Rope]'s elements.
#[derive(Debug, Clone)]
pub struct Iter<'a, M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    chunks: Chunks<'a, M>,
    cur_chunk: &'a [M],
    index: usize,
    width: usize,
    last_call_was_prev_impl: bool,
    total_len: usize,
    remaining_len: usize,
    is_reversed: bool,
}

impl<'a, M> Iter<'a, M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    pub(crate) fn new(node: &'a Arc<Node<M>>) -> Self {
        let mut chunk_iter = Chunks::new(node);
        let cur_chunk = if let Some(chunk) = chunk_iter.next() {
            chunk
        } else {
            &[]
        };
        Iter {
            chunks: chunk_iter,
            cur_chunk,
            index: 0,
            width: 0,
            last_call_was_prev_impl: false,
            total_len: node.slice_info().len as usize,
            remaining_len: node.slice_info().len as usize,
            is_reversed: false,
        }
    }

    #[inline(always)]
    pub(crate) fn new_with_range(
        node: &'a Arc<Node<M>>,
        index_range: (usize, usize),
        width_range: (usize, usize),
    ) -> Self {
        Iter::new_with_range_at_width(node, width_range.0, index_range, width_range)
    }

    pub(crate) fn new_with_range_at_width(
        node: &'a Arc<Node<M>>,
        at_width: usize,
        index_range: (usize, usize),
        width_range: (usize, usize),
    ) -> Self {
        let (mut chunks, mut chunk_start_index, mut chunk_start_width) =
            Chunks::new_with_range_at_width(node, at_width, index_range, width_range);

        let cur_chunk = if index_range.0 == index_range.1 {
            &[]
        } else if let Some(chunk) = chunks.next() {
            chunk
        } else {
            let chunk = chunks.prev().unwrap();
            chunks.next();
            chunk_start_index -= chunk.len();
            chunk_start_width -= chunk.iter().map(|m| m.width()).sum::<usize>();
            chunk
        };

        let index = start_width_to_index(cur_chunk, at_width - chunk_start_width);
        let width = index_to_width(cur_chunk, index) + chunk_start_width;

        Iter {
            chunks,
            cur_chunk,
            index,
            width,
            last_call_was_prev_impl: false,
            total_len: index_range.1 - index_range.0,
            remaining_len: index_range.1 - (index + chunk_start_index),
            is_reversed: false,
        }
    }

    #[inline(always)]
    pub(crate) fn from_slice(slice: &'a [M]) -> Self {
        Iter::from_slice_at(slice, 0)
    }

    pub(crate) fn from_slice_at(slice: &'a [M], width: usize) -> Self {
        let mut chunks = Chunks::from_slice(slice, false);
        let cur_chunk = if let Some(chunk) = chunks.next() {
            chunk
        } else {
            &[]
        };

        let index = start_width_to_index(slice, width);
        let width = index_to_width(slice, index);

        Iter {
            chunks,
            cur_chunk,
            index,
            width,
            last_call_was_prev_impl: false,
            total_len: slice.len(),
            remaining_len: slice.len() - index,
            is_reversed: false,
        }
    }

    /// Reverses the direction of the iterator in-place.
    ///
    /// In other words, swaps the behavior of [`prev()`][Self::prev]
    /// and [`next()`][Self::next].
    #[inline]
    pub fn reverse(&mut self) {
        self.is_reversed = !self.is_reversed;
    }

    /// Same as [`reverse()`][Self::reverse], but returns itself.
    ///
    /// This is useful when chaining iterator methods:
    ///
    /// ```rust
    /// # use any_rope::{Rope, Width};
    /// # let rope = Rope::from_slice(&[Width(1), Width(2), Width(5)]);
    /// // Print the rope's elements and their widths in reverse.
    /// for (width, element) in rope.iter_at_width(rope.width()).reversed() {
    ///     println!("{} {:?}", width, element);
    /// #   assert_eq!((width, element), rope.from_width(width));
    /// }
    #[inline]
    #[must_use]
    pub fn reversed(mut self) -> Self {
        self.reverse();
        self
    }

    /// Advances the iterator backwards and returns the previous value.
    ///
    /// Runs in amortized O(1) time and worst-case O(log N) time.
    #[inline(always)]
    pub fn prev(&mut self) -> Option<(usize, M)> {
        if !self.is_reversed {
            self.prev_impl()
        } else {
            self.next_impl()
        }
    }

    #[inline]
    fn prev_impl(&mut self) -> Option<(usize, M)> {
        // Put us back into a "prev" progression.
        if !self.last_call_was_prev_impl {
            self.chunks.prev();
            self.last_call_was_prev_impl = true;
        }

        // Progress the chunks iterator back if needed.
        if self.index == 0 {
            if let Some(chunk) = self.chunks.prev() {
                self.cur_chunk = chunk;
                self.index = self.cur_chunk.len();
            } else {
                return None;
            }
        }

        // Progress the byte counts and return the previous element.
        self.index -= 1;
        self.remaining_len += 1;
        self.width -= self.cur_chunk[self.index].width();
        return Some((self.width, self.cur_chunk[self.index]));
    }

    #[inline]
    fn next_impl(&mut self) -> Option<(usize, M)> {
        // Put us back into a "next" progression.
        if self.last_call_was_prev_impl {
            self.chunks.next();
            self.last_call_was_prev_impl = false;
        }

        // Progress the chunks iterator forward if needed.
        if self.index >= self.cur_chunk.len() {
            if let Some(chunk) = self.chunks.next() {
                self.cur_chunk = chunk;
                self.index = 0;
            } else {
                return None;
            }
        }

        // Progress the byte counts and return the next element.
        let element = self.cur_chunk[self.index];
        self.index += 1;
        self.remaining_len -= 1;

        let old_width = self.width;
        self.width += element.width();
        return Some((old_width, element));
    }
}

impl<'a, M> Iterator for Iter<'a, M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    type Item = (usize, M);

    /// Advances the iterator forward and returns the next value.
    ///
    /// Runs in amortized O(1) time and worst-case O(log N) time.
    #[inline(always)]
    fn next(&mut self) -> Option<(usize, M)> {
        if !self.is_reversed {
            self.next_impl()
        } else {
            self.prev_impl()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if !self.is_reversed {
            self.remaining_len
        } else {
            self.total_len - self.remaining_len
        };
        (remaining, Some(remaining))
    }
}

impl<'a, M> ExactSizeIterator for Iter<'a, M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
}

//==========================================================

/// An iterator over a [`Rope<M>`]'s contiguous [`M`] chunks.
///
/// Internally, each [`Rope<M>`] stores [`M`]s as a segemented collection of
/// [`&[M]`][Measurable]. It is useful for situations such as:
///
/// - Streaming a [`Rope<M>`]'s elements data somewhere.
/// - Writing custom iterators over a [`Rope<M>`]'s data.
///
/// There are no guarantees about the size of yielded chunks, and there are
/// no guarantees about where the chunks are split.  For example, they may
/// be zero-sized.
///
/// [`M`]: Measurable
/// [`Rope<M>`]: crate::rope::Rope
#[derive(Debug, Clone)]
pub struct Chunks<'a, M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    iter: ChunksEnum<'a, M>,
    is_reversed: bool,
}

#[derive(Debug, Clone)]
enum ChunksEnum<'a, M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    Full {
        /// (node ref, index of current child)
        node_stack: Vec<(&'a Arc<Node<M>>, usize)>,
        /// Total lenght of the data range of the iterator.
        len: usize,
        /// The index of the current element relative to the data range start.
        index: isize,
    },
    Light {
        slice: &'a [M],
        is_end: bool,
    },
}

impl<'a, M> Chunks<'a, M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    #[inline(always)]
    pub(crate) fn new(node: &'a Arc<Node<M>>) -> Self {
        let info = node.slice_info();
        Chunks::new_with_range_at_index(node, 0, (0, info.len as usize), (0, info.width as usize)).0
    }

    #[inline(always)]
    pub(crate) fn new_with_range(
        node: &'a Arc<Node<M>>,
        index_range: (usize, usize),
        width_range: (usize, usize),
    ) -> Self {
        Chunks::new_with_range_at_index(node, index_range.0, index_range, width_range).0
    }

    /// The main workhorse function for creating new [`Chunks`] iterators.
    ///
    /// Creates a new [`Chunks`] iterator from the given node, starting the
    /// iterator at the chunk containing the element in `at_index`
    /// (i.e. the [`next()`][Self::next] method will yield the chunk containing
    /// that element). The range of the iterator is bounded by `index_range`.
    ///
    /// Both `at_index` and `index_range` are relative to the beginning of
    /// of the passed node.
    ///
    /// Passing an `at_index` equal to the max of `index_range` creates an
    /// iterator at the end of forward iteration.
    ///
    /// Returns the iterator and the index/width of its start relative
    /// to the start of the node.
    pub(crate) fn new_with_range_at_index(
        node: &Arc<Node<M>>,
        at_index: usize,
        index_range: (usize, usize),
        width_range: (usize, usize),
    ) -> (Chunks<M>, usize, usize) {
        debug_assert!(at_index >= index_range.0);
        debug_assert!(at_index <= index_range.1);

        // For convenience/readability.
        let start_index = index_range.0;
        let end_index = index_range.1;

        // Special-case for empty slice contents.
        if start_index == end_index {
            return (
                Chunks {
                    iter: ChunksEnum::Light {
                        slice: &[],
                        is_end: false,
                    },
                    is_reversed: false,
                },
                0,
                0,
            );
        }

        // If root is a leaf, return light version of the iter.
        if node.is_leaf() {
            let slice = &node.leaf_slice()[start_index..end_index];
            if at_index == end_index {
                return (
                    Chunks {
                        iter: ChunksEnum::Light {
                            slice,
                            is_end: true,
                        },
                        is_reversed: false,
                    },
                    slice.len(),
                    width_of(slice),
                );
            } else {
                return (
                    Chunks {
                        iter: ChunksEnum::Light {
                            slice,
                            is_end: false,
                        },
                        is_reversed: false,
                    },
                    0,
                    0,
                );
            }
        }

        // Create and populate the node stack, and determine the char index
        // within the first chunk, and byte index of the start of that chunk.
        let mut info = SliceInfo::new();
        let mut index = at_index as isize;
        let node_stack = {
            let mut node_stack: Vec<(&Arc<Node<M>>, usize)> = Vec::new();
            let mut node_ref = node;
            loop {
                match **node_ref {
                    Node::Leaf(ref slice) => {
                        if at_index < end_index || index == 0 {
                            index = info.len as isize - start_index as isize;
                        } else {
                            index =
                                (info.len as isize + slice.len() as isize) - start_index as isize;
                            info = SliceInfo {
                                len: index_range.1 as u64,
                                width: width_range.1 as u64,
                            };
                            node_stack.last_mut().unwrap().1 += 1;
                        }
                        break;
                    }
                    Node::Branch(ref children) => {
                        let (child_i, acc_info) = children.search_index(index as usize);
                        info += acc_info;
                        node_stack.push((node_ref, child_i));
                        node_ref = &children.nodes()[child_i];
                        index -= acc_info.len as isize;
                    }
                }
            }
            node_stack
        };

        // Create the iterator.
        (
            Chunks {
                iter: ChunksEnum::Full {
                    node_stack,
                    len: end_index - start_index,
                    index,
                },
                is_reversed: false,
            },
            (info.len as usize).max(index_range.0),
            (info.width as usize).max(width_range.0),
        )
    }

    #[inline(always)]
    pub(crate) fn new_with_range_at_width(
        node: &'a Arc<Node<M>>,
        at_width: usize,
        index_range: (usize, usize),
        width_range: (usize, usize),
    ) -> (Self, usize, usize) {
        let at_index = (node.get_first_chunk_at_width(at_width).1.len as usize).max(index_range.0);

        Chunks::new_with_range_at_index(node, at_index, index_range, width_range)
    }

    pub(crate) fn from_slice(slice: &'a [M], is_end: bool) -> Self {
        Chunks {
            iter: ChunksEnum::Light { slice, is_end },
            is_reversed: false,
        }
    }

    /// Reverses the direction of the iterator in-place.
    ///
    /// In other words, swaps the behavior of [`prev()`][Self::prev]
    /// and [`next()`][Self::next].
    #[inline]
    pub fn reverse(&mut self) {
        self.is_reversed = !self.is_reversed;
    }

    /// Same as [`reverse()`][Self::reverse], but returns itself.
    ///
    /// This is useful when chaining iterator methods:
    ///
    /// ```rust
    /// # use any_rope::{Rope, Width};
    /// # let rope = Rope::from_slice(
    /// #    &[Width(1), Width(2), Width(3), Width(0), Width(0)]
    /// # );
    /// // Enumerate the rope's chunks in reverse, starting from the end.
    /// for (index, chunk) in rope.chunks_at_index(rope.len()).0.reversed().enumerate() {
    ///     println!("{} {:?}", index, chunk);
    /// #   assert_eq!(chunk, rope.chunks().nth(rope.chunks().count() - index - 1).unwrap());
    /// }
    #[inline]
    #[must_use]
    pub fn reversed(mut self) -> Self {
        self.reverse();
        self
    }

    /// Advances the iterator backwards and returns the previous value.
    ///
    /// Runs in amortized O(1) time and worst-case O(log N) time.
    #[inline(always)]
    pub fn prev(&mut self) -> Option<&'a [M]> {
        if !self.is_reversed {
            self.prev_impl()
        } else {
            self.next_impl()
        }
    }

    fn prev_impl(&mut self) -> Option<&'a [M]> {
        match *self {
            Chunks {
                iter:
                    ChunksEnum::Full {
                        ref mut node_stack,
                        len,
                        ref mut index,
                    },
                ..
            } => {
                if *index <= 0 {
                    return None;
                }

                // Progress the node stack if needed.
                let mut stack_index = node_stack.len() - 1;
                if node_stack[stack_index].1 == 0 {
                    while node_stack[stack_index].1 == 0 {
                        if stack_index == 0 {
                            return None;
                        } else {
                            stack_index -= 1;
                        }
                    }
                    node_stack[stack_index].1 -= 1;
                    while stack_index < (node_stack.len() - 1) {
                        let child_i = node_stack[stack_index].1;
                        let node = &node_stack[stack_index].0.children().nodes()[child_i];
                        node_stack[stack_index + 1] = (node, node.child_count() - 1);
                        stack_index += 1;
                    }
                    node_stack[stack_index].1 += 1;
                }

                // Fetch the node and child index.
                let (node, ref mut child_i) = node_stack.last_mut().unwrap();
                *child_i -= 1;

                // Get the slice in the appropriate range.
                let slice = node.children().nodes()[*child_i].leaf_slice();
                *index -= slice.len() as isize;
                let slice = {
                    let start_byte = if *index < 0 { (-*index) as usize } else { 0 };
                    let end_byte = slice.len().min((len as isize - *index) as usize);
                    &slice[start_byte..end_byte]
                };

                // Return the slice.
                return Some(slice);
            }

            Chunks {
                iter:
                    ChunksEnum::Light {
                        slice,
                        ref mut is_end,
                    },
                ..
            } => {
                if !*is_end || slice.is_empty() {
                    return None;
                } else {
                    *is_end = false;
                    return Some(slice);
                }
            }
        }
    }

    fn next_impl(&mut self) -> Option<&'a [M]> {
        match *self {
            Chunks {
                iter:
                    ChunksEnum::Full {
                        ref mut node_stack,
                        len,
                        ref mut index,
                    },
                ..
            } => {
                if *index >= len as isize {
                    return None;
                }

                // Progress the node stack if needed.
                let mut stack_index = node_stack.len() - 1;
                if node_stack[stack_index].1 >= node_stack[stack_index].0.child_count() {
                    while node_stack[stack_index].1 >= (node_stack[stack_index].0.child_count() - 1)
                    {
                        if stack_index == 0 {
                            return None;
                        } else {
                            stack_index -= 1;
                        }
                    }
                    node_stack[stack_index].1 += 1;
                    while stack_index < (node_stack.len() - 1) {
                        let child_i = node_stack[stack_index].1;
                        let node = &node_stack[stack_index].0.children().nodes()[child_i];
                        node_stack[stack_index + 1] = (node, 0);
                        stack_index += 1;
                    }
                }

                // Fetch the node and child index.
                let (node, ref mut child_i) = node_stack.last_mut().unwrap();

                // Get the slice, sliced to the appropriate range.
                let leaf_slice = node.children().nodes()[*child_i].leaf_slice();
                let slice = {
                    let start_byte = if *index < 0 { (-*index) as usize } else { 0 };
                    let end_byte = leaf_slice.len().min((len as isize - *index) as usize);
                    &leaf_slice[start_byte..end_byte]
                };

                // Book keeping.
                *index += leaf_slice.len() as isize;
                *child_i += 1;

                // Return the slice.
                return Some(slice);
            }

            Chunks {
                iter:
                    ChunksEnum::Light {
                        slice,
                        ref mut is_end,
                    },
                ..
            } => {
                if *is_end || slice.is_empty() {
                    return None;
                } else {
                    *is_end = true;
                    return Some(slice);
                }
            }
        }
    }
}

impl<'a, M> Iterator for Chunks<'a, M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    type Item = &'a [M];

    /// Advances the iterator forward and returns the next value.
    ///
    /// Runs in amortized O(1) time and worst-case O(log N) time.
    #[inline(always)]
    fn next(&mut self) -> Option<&'a [M]> {
        if !self.is_reversed {
            self.next_impl()
        } else {
            self.prev_impl()
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::while_let_on_iterator)]
    use super::*;
    use crate::{Rope, Width};

    fn pseudo_random() -> Vec<Width> {
        (0..1400)
            .map(|num| match num % 14 {
                0 => Width(1),
                1 => Width(2),
                2 => Width(4),
                3 => Width(0),
                4 => Width(0),
                5 => Width(5),
                6 => Width(1),
                7 => Width(1),
                8 => Width(2),
                9 => Width(8),
                10 => Width(0),
                11 => Width(0),
                12 => Width(3),
                13 => Width(0),
                _ => unreachable!(),
            })
            .collect()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        for ((_, from_rope), from_vec) in rope.iter().zip(pseudo_random().iter().copied()) {
            assert_eq!(from_rope, from_vec);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let mut iter = rope.iter();
        while let Some(_) = iter.next() {}

        let mut i = pseudo_random().len();
        while let Some((_, element)) = iter.prev() {
            i -= 1;
            assert_eq!(element, pseudo_random()[i]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_03() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let mut iter = rope.iter();

        iter.next();
        iter.prev();
        assert_eq!(None, iter.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_04() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let mut iter = rope.iter();
        while let Some(_) = iter.next() {}

        iter.prev();
        iter.next();
        assert_eq!(None, iter.next());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_05() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let mut iter = rope.iter();

        assert_eq!(None, iter.prev());
        iter.next();
        iter.prev();
        assert_eq!(None, iter.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_06() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let mut iter = rope.iter();
        while let Some(_) = iter.next() {}

        assert_eq!(None, iter.next());
        iter.prev();
        iter.next();
        assert_eq!(None, iter.next());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_07() {
        let mut iter = Iter::from_slice(&[Width(1)]);

        assert_eq!(Some((0, Width(1))), iter.next());
        assert_eq!(None, iter.next());
        assert_eq!(Some((0, Width(1))), iter.prev());
        assert_eq!(None, iter.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_08() {
        let width_vec = pseudo_random();
        let mut iter = Iter::from_slice(width_vec.as_slice());

        assert_eq!(iter.next(), Some((0, Width(1))));
        assert_eq!(iter.next(), Some((1, Width(2))));
        assert_eq!(iter.next(), Some((3, Width(4))));
        assert_eq!(iter.next(), Some((7, Width(0))));
        assert_eq!(iter.next(), Some((7, Width(0))));
        assert_eq!(iter.next(), Some((7, Width(5))));
        assert_eq!(iter.next(), Some((12, Width(1))));
        assert_eq!(iter.next(), Some((13, Width(1))));
        assert_eq!(iter.next(), Some((14, Width(2))));
        assert_eq!(iter.next(), Some((16, Width(8))));
        assert_eq!(iter.next(), Some((24, Width(0))));
        assert_eq!(iter.next(), Some((24, Width(0))));
        assert_eq!(iter.next(), Some((24, Width(3))));
        assert_eq!(iter.next(), Some((27, Width(0))));
        assert_eq!(iter.next(), Some((27, Width(1))));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_at_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.width_slice(..79);
        let mut iter = slice.iter_at(56);

        assert_eq!(iter.next(), Some((55, Width(2))));
        assert_eq!(iter.next(), Some((57, Width(4))));
        assert_eq!(iter.next(), Some((61, Width(0))));
        assert_eq!(iter.next(), Some((61, Width(0))));
        assert_eq!(iter.next(), Some((61, Width(5))));
        assert_eq!(iter.next(), Some((66, Width(1))));
        assert_eq!(iter.next(), Some((67, Width(1))));
        assert_eq!(iter.next(), Some((68, Width(2))));
        assert_eq!(iter.next(), Some((70, Width(8))));
        assert_eq!(iter.next(), Some((78, Width(0))));
        assert_eq!(iter.next(), Some((78, Width(0))));
        assert_eq!(iter.next(), Some((78, Width(3))));
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_at_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let mut bytes = rope.iter_at_width(rope.width());
        // Iterating at the end, when there are zero width elements, always yields them.
        assert_eq!(bytes.next(), Some((2700, Width(0))));
        assert_eq!(bytes.next(), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_at_03() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let mut iter_1 = rope.iter_at_width(rope.width());
        let width_vec = pseudo_random();
        // Skip the last element, since it's zero width.
        let mut iter_2 = width_vec.iter().take(1399).copied();

        while let Some(b) = iter_2.next_back() {
            assert_eq!(iter_1.prev().map(|(_, element)| element), Some(b));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn exact_size_iter_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.width_slice(34..75);

        let mut len = slice.len();
        let mut iter = slice.iter();
        assert_eq!(len, iter.len());

        while let Some(_) = iter.next() {
            len -= 1;
            assert_eq!(len, iter.len());
        }

        iter.next();
        iter.next();
        iter.next();
        iter.next();
        iter.next();
        iter.next();
        iter.next();
        assert_eq!(iter.len(), 0);
        assert_eq!(len, 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn exact_size_iter_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.width_slice(34..300);

        let mut len = 0;
        let mut iter = slice.iter_at(slice.width());

        assert_eq!(len, iter.len());

        while iter.prev().is_some() {
            len += 1;
            assert_eq!(len, iter.len());
        }

        assert_eq!(iter.len(), slice.len());
        iter.prev();
        iter.prev();
        iter.prev();
        iter.prev();
        iter.prev();
        iter.prev();
        iter.prev();
        assert_eq!(iter.len(), slice.len());
        assert_eq!(len, slice.len());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn exact_size_iter_03() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.width_slice(34..34);
        let mut iter = slice.iter();

        assert_eq!(iter.next(), Some((34, Width(0))));
        assert_eq!(iter.next(), Some((34, Width(0))));
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_reverse_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let mut iter = rope.iter();
        let mut stack = Vec::new();

        for _ in 0..32 {
            stack.push(iter.next().unwrap());
        }
        iter.reverse();
        for _ in 0..32 {
            assert_eq!(stack.pop(), iter.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_reverse_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let mut iter = rope.iter_at_width(rope.len() / 3);
        let mut stack = Vec::new();

        for _ in 0..32 {
            stack.push(iter.next().unwrap());
        }
        iter.reverse();
        for _ in 0..32 {
            assert_eq!(stack.pop(), iter.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());

        let mut index = 0;
        for chunk in rope.chunks() {
            assert_eq!(chunk, &pseudo_random()[index..(index + chunk.len())]);
            index += chunk.len();
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_02() {
        let rope = Rope::<Width>::from_slice(&[]);
        let mut iter = rope.chunks();

        assert_eq!(None, iter.next());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_03() {
        let rope = Rope::from_slice(pseudo_random().as_slice());

        let mut iter = rope.chunks();

        assert_eq!(None, iter.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_04() {
        let rope = Rope::from_slice(pseudo_random().as_slice());

        let mut chunks = Vec::new();
        let mut iter = rope.chunks();

        while let Some(slice) = iter.next() {
            chunks.push(slice);
        }

        while let Some(slice) = iter.prev() {
            assert_eq!(slice, chunks.pop().unwrap());
        }

        assert!(chunks.is_empty());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_at_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());

        for i in 0..rope.len() {
            let (chunk, index, width) = rope.chunk_at_index(i);
            let (mut chunks, slice_index, slice_width) = rope.chunks_at_index(i);

            assert_eq!(index, slice_index);
            assert_eq!(width, slice_width);
            assert_eq!(Some(chunk), chunks.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_at_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.width_slice(34..301);

        let (mut chunks, ..) = slice.chunks_at_index(slice.len());
        assert_eq!(chunks.next(), None);

        let (mut chunks, ..) = slice.chunks_at_index(slice.len());
        assert_eq!(slice.chunks().last(), chunks.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_at_03() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.width_slice(34..34);

        let (mut chunks, ..) = slice.chunks_at_index(0);
        assert_eq!(chunks.next(), Some([Width(0)].as_slice()));
        assert!(chunks.next().is_some());

        let (mut chunks, ..) = slice.chunks_at_index(0);
        assert_eq!(chunks.prev(), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_reverse_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let mut iter = rope.chunks();
        let mut stack = Vec::new();

        for _ in 0..8 {
            stack.push(iter.next().unwrap());
        }
        iter.reverse();
        for _ in 0..8 {
            assert_eq!(stack.pop().unwrap(), iter.next().unwrap());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_reverse_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let mut iter = rope.chunks_at_width(rope.width() / 3).0;
        let mut stack = Vec::new();

        for _ in 0..8 {
            stack.push(iter.next().unwrap());
        }
        iter.reverse();
        for _ in 0..8 {
            assert_eq!(stack.pop().unwrap(), iter.next().unwrap());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_reverse_03() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let mut iter = rope.chunks_at_width(rope.width() / 3).0;
        let mut stack = Vec::new();

        iter.reverse();
        for _ in 0..8 {
            stack.push(iter.next().unwrap());
        }
        iter.reverse();
        for _ in 0..8 {
            assert_eq!(stack.pop().unwrap(), iter.next().unwrap());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_reverse_04() {
        let mut iter = Chunks::from_slice(&[Width(5), Width(0)], false);

        assert_eq!(Some([Width(5), Width(0)].as_slice()), iter.next());
        assert_eq!(None, iter.next());
        iter.reverse();
        assert_eq!(Some([Width(5), Width(0)].as_slice()), iter.next());
        assert_eq!(None, iter.next());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_sliced_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());

        let slice_start = 34;
        let slice_end = 301;
        let slice_start_byte = rope.start_width_to_index(slice_start);
        let s_end_byte = rope.end_width_to_index(slice_end);

        let slice_1 = rope.width_slice(slice_start..slice_end);
        let slice_2 = &pseudo_random()[slice_start_byte..s_end_byte];

        let mut slice_1_iter = slice_1.iter();
        let mut slice_2_iter = slice_2.iter().copied();

        assert_eq!(slice_1, slice_2);
        assert_eq!(slice_1.from_index(0).1, slice_2[0]);
        for _ in 0..(slice_2.len() + 1) {
            assert_eq!(
                slice_1_iter.next().map(|(_, element)| element),
                slice_2_iter.next()
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_at_sliced_02() {
        let rope = Rope::from_slice(pseudo_random().as_slice());
        let slice = rope.width_slice(34..300);
        let mut iter = slice.iter_at(slice.width());
        // Yields None, since we're iterating in the middle of a Width(4) element.
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_at_sliced_03() {
        let rope = Rope::from_slice(pseudo_random().as_slice());

        let slice_start = 34;
        let slice_end = 300;
        let slice_start_byte = rope.start_width_to_index(slice_start);
        let s_end_byte = rope.end_width_to_index(slice_end);

        let slice_1 = rope.width_slice(slice_start..slice_end);
        let slice_2 = &pseudo_random()[slice_start_byte..s_end_byte];

        let mut bytes_1 = slice_1.iter_at(slice_1.width());
        let mut bytes_2 = slice_2.iter().copied();
        while let Some(b) = bytes_2.next_back() {
            assert_eq!(bytes_1.prev().map(|(_, element)| element), Some(b));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_at_sliced_reverse_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());

        let slice_start = 34;
        let slice_end = 301;
        let slice = rope.width_slice(slice_start..slice_end);

        let mut iter = slice.iter_at(slice.len() / 3);
        let mut stack = Vec::new();
        for _ in 0..32 {
            stack.push(iter.next().unwrap());
        }
        iter.reverse();
        for _ in 0..32 {
            assert_eq!(stack.pop(), iter.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_sliced_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());

        let slice_start = 34;
        let slice_end = 301;
        let slice_start_index = rope.start_width_to_index(slice_start);
        let slice_end_index = rope.end_width_to_index(slice_end);

        let slice_1 = rope.width_slice(slice_start..slice_end);
        let slice_2 = &pseudo_random()[slice_start_index..slice_end_index];

        let mut index = 0;
        for chunk in slice_1.chunks() {
            assert_eq!(chunk, &slice_2[index..(index + chunk.len())]);
            index += chunk.len();
        }

        assert_eq!(index, slice_2.len());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_sliced_reverse_01() {
        let rope = Rope::from_slice(pseudo_random().as_slice());

        let slice_start = 34;
        let slice_end = 301;
        let slice = rope.width_slice(slice_start..slice_end);

        let mut iter = slice.chunks();
        let mut stack = Vec::new();
        for _ in 0..8 {
            stack.push(iter.next().unwrap());
        }
        iter.reverse();
        for _ in 0..8 {
            assert_eq!(stack.pop(), iter.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn empty_iter() {
        let rope = Rope::<Width>::from_slice(&[]);
        let rope: Vec<Width> = rope.iter().map(|(_, element)| element).collect();
        assert_eq!(&*rope, [].as_slice())
    }
}
