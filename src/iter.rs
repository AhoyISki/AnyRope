//! Iterators over a `Rope`'s data.
//!
//! The iterators in Ropey can be created from both `Rope`s and `RopeSlice`s.
//! When created from a `RopeSlice`, they iterate over only the data that the
//! `RopeSlice` refers to.  For the `Lines` and `Chunks` iterators, the data
//! of the first and last yielded item will be correctly truncated to match
//! the bounds of the `RopeSlice`.
//!
//! # Reverse iteration
//!
//! All iterators in Ropey operate as a cursor that can move both forwards
//! and backwards over its contents.  This can be accomplished via the
//! `next()` and `prev()` methods on each iterator, or by using the `reverse()`
//! or `reversed()` methods to change the iterator's direction.
//!
//! Conceptually, an iterator in Ropey is always positioned *between* the
//! elements it iterates over, and returns an element when it jumps over it
//! via the `next()` or `prev()` methods.
//!
//! For example, given the text `"abc"` and a `Chars` iterator starting at the
//! beginning of the text, you would get the following sequence of states and
//! return values by repeatedly calling `next()` (the vertical bar represents
//! the position of the iterator):
//!
//! 0. `|abc`
//! 1. `a|bc` -> `Some('a')`
//! 2. `ab|c` -> `Some('b')`
//! 3. `abc|` -> `Some('c')`
//! 4. `abc|` -> `None`
//!
//! The `prev()` method operates identically, except moving in the opposite
//! direction.  And `reverse()` simply swaps the behavior of `prev()` and
//! `next()`.
//!
//! # Creating iterators at any position
//!
//! Iterators in Ropey can be created starting at any position in the text.
//! This is accomplished with the various `bytes_at()`, `chars_at()`, etc.
//! methods of `Rope` and `RopeSlice`.
//!
//! When an iterator is created this way, it is positioned such that a call to
//! `next()` will return the specified element, and a call to `prev()` will
//! return the element just before the specified one.
//!
//! Importantly, iterators created this way still have access to the entire
//! contents of the `Rope`/`RopeSlice` they were created from&mdash;the
//! contents before the specified position is not truncated.  For example, you
//! can create a `Chars` iterator starting at the end of a `Rope`, and then
//! use the `prev()` method to iterate backwards over all of that `Rope`'s
//! chars.
//!
//! # A possible point of confusion
//!
//! The Rust standard library has an iterator trait `DoubleEndedIterator` with
//! a method `rev()`.  While this method's name is very similar to Ropey's
//! `reverse()` method, its behavior is very different.
//!
//! `DoubleEndedIterator` actually provides two iterators: one starting at each
//! end of the collection, moving in opposite directions towards each other.
//! Calling `rev()` switches between those two iterators, changing not only the
//! direction of iteration but also its current position in the collection.
//!
//! The `reverse()` method on Ropey's iterators, on the other hand, reverses
//! the direction of the iterator in-place, without changing its position in
//! the text.

use std::sync::Arc;

use crate::rope::Measurable;
use crate::slice_utils::width_of;
use crate::tree::{Node, SliceInfo};

//==========================================================

/// An iterator over a `Rope`'s bytes.
#[derive(Debug, Clone)]
pub struct Iter<'a, M>
where
    M: Measurable,
{
    chunk_iter: Chunks<'a, M>,
    cur_chunk: &'a [M],
    index: usize,
    last_call_was_prev_impl: bool,
    total_len: usize,
    remaining_len: usize,
    is_reversed: bool,
}

impl<'a, M> Iter<'a, M>
where
    M: Measurable,
{
    pub(crate) fn new(node: &'a Arc<Node<M>>) -> Self {
        let mut chunk_iter = Chunks::new(node);
        let cur_chunk = if let Some(chunk) = chunk_iter.next() {
            chunk
        } else {
            &[]
        };
        Iter {
            chunk_iter: chunk_iter,
            cur_chunk,
            index: 0,
            last_call_was_prev_impl: false,
            total_len: node.slice_info().len as usize,
            remaining_len: node.slice_info().len as usize,
            is_reversed: false,
        }
    }

    #[inline(always)]
    pub(crate) fn new_with_range(
        node: &'a Arc<Node<M>>,
        byte_index_range: (usize, usize),
        width_index_range: (usize, usize),
    ) -> Self {
        Iter::new_with_range_at(
            node,
            byte_index_range.0,
            byte_index_range,
            width_index_range,
        )
    }

    pub(crate) fn new_with_range_at(
        node: &'a Arc<Node<M>>,
        at_index: usize,
        index_range: (usize, usize),
        width_range: (usize, usize),
    ) -> Self {
        let (mut chunk_iter, mut chunk_start_index, _) =
            Chunks::new_with_range_at_index(node, at_index, index_range, width_range);

        let cur_chunk = if index_range.0 == index_range.1 {
            &[]
        } else if at_index < index_range.1 {
            chunk_iter.next().unwrap()
        } else {
            let chunk = chunk_iter.prev().unwrap();
            chunk_iter.next();
            chunk_start_index -= chunk.len();
            chunk
        };
        println!(
            "{:?}, {:?}, {:?}, {:?}, {:?}",
            index_range, at_index, chunk_start_index, width_range, cur_chunk
        );

        Iter {
            chunk_iter,
            cur_chunk,
            index: at_index - chunk_start_index,
            last_call_was_prev_impl: false,
            total_len: index_range.1 - index_range.0,
            remaining_len: index_range.1 - at_index,
            is_reversed: false,
        }
    }

    #[inline(always)]
    pub(crate) fn from_slice(slice: &'a [M]) -> Self {
        Iter::from_slice_at(slice, 0)
    }

    pub(crate) fn from_slice_at(slice: &'a [M], index: usize) -> Self {
        let mut chunk_iter = Chunks::from_slice(slice, false);
        let cur_chunk = if let Some(chunk) = chunk_iter.next() {
            chunk
        } else {
            &[]
        };
        Iter {
            chunk_iter: chunk_iter,
            cur_chunk: cur_chunk,
            index,
            last_call_was_prev_impl: false,
            total_len: slice.len(),
            remaining_len: slice.len() - index,
            is_reversed: false,
        }
    }

    /// Reverses the direction of the iterator in-place.
    ///
    /// In other words, swaps the behavior of [`prev()`](Bytes::prev())
    /// and [`next()`](Bytes::next()).
    #[inline]
    pub fn reverse(&mut self) {
        self.is_reversed = !self.is_reversed;
    }

    /// Same as `reverse()`, but returns itself.
    ///
    /// This is useful when chaining iterator methods:
    ///
    /// ```rust
    /// # use ropey::Rope;
    /// # let rope = Rope::from_str("Hello there\n world!\n");
    /// // Enumerate the rope's bytes in reverse, starting from the end.
    /// for (i, b) in rope.bytes_at(rope.len_bytes()).reversed().enumerate() {
    ///     println!("{} {}", i, b);
    /// #   assert_eq!(b, rope.byte(rope.len_bytes() - i - 1));
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
    pub fn prev(&mut self) -> Option<M> {
        if !self.is_reversed {
            self.prev_impl()
        } else {
            self.next_impl()
        }
    }

    #[inline]
    fn prev_impl(&mut self) -> Option<M> {
        // Put us back into a "prev" progression.
        if !self.last_call_was_prev_impl {
            self.chunk_iter.prev();
            self.last_call_was_prev_impl = true;
        }

        // Progress the chunks iterator back if needed.
        if self.index == 0 {
            if let Some(chunk) = self.chunk_iter.prev() {
                self.cur_chunk = chunk;
                self.index = self.cur_chunk.len();
            } else {
                return None;
            }
        }

        // Progress the byte counts and return the previous byte.
        self.index -= 1;
        self.remaining_len += 1;
        return Some(self.cur_chunk[self.index]);
    }

    #[inline]
    fn next_impl(&mut self) -> Option<M> {
        // Put us back into a "next" progression.
        if self.last_call_was_prev_impl {
            self.chunk_iter.next();
            self.last_call_was_prev_impl = false;
        }

        // Progress the chunks iterator forward if needed.
        if self.index >= self.cur_chunk.len() {
            if let Some(chunk) = self.chunk_iter.next() {
                self.cur_chunk = chunk;
                self.index = 0;
            } else {
                return None;
            }
        }

        // Progress the byte counts and return the next byte.
        let index = self.cur_chunk[self.index];
        self.index += 1;
        self.remaining_len -= 1;
        return Some(index);
    }
}

impl<'a, M> Iterator for Iter<'a, M>
where
    M: Measurable,
{
    type Item = M;

    /// Advances the iterator forward and returns the next value.
    ///
    /// Runs in amortized O(1) time and worst-case O(log N) time.
    #[inline(always)]
    fn next(&mut self) -> Option<M> {
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

impl<'a, M> ExactSizeIterator for Iter<'a, M> where M: Measurable {}

//==========================================================

/// An iterator over a `Rope`'s contiguous `str` chunks.
///
/// Internally, each `Rope` stores text as a segemented collection of utf8
/// strings. This iterator iterates over those segments, returning a
/// `&str` slice for each one.  It is useful for situations such as:
///
/// - Writing a rope's utf8 text data to disk (but see
///   [`write_to()`](crate::rope::Rope::write_to) for a convenience function that does this
///   for casual use-cases).
/// - Streaming a rope's text data somewhere.
/// - Saving a rope to a non-utf8 encoding, doing the encoding conversion
///   incrementally as you go.
/// - Writing custom iterators over a rope's text data.
///
/// There are precisely two guarantees about the yielded chunks:
///
/// - All non-empty chunks are yielded, and they are yielded in order.
/// - CRLF pairs are never split across chunks.
///
/// There are no guarantees about the size of yielded chunks, and except for
/// CRLF pairs and being valid `str` slices there are no guarantees about
/// where the chunks are split.  For example, they may be zero-sized, they
/// don't necessarily align with line breaks, etc.
#[derive(Debug, Clone)]
pub struct Chunks<'a, M>
where
    M: Measurable,
{
    iter: ChunksEnum<'a, M>,
    is_reversed: bool,
}

#[derive(Debug, Clone)]
enum ChunksEnum<'a, M>
where
    M: Measurable,
{
    Full {
        node_stack: Vec<(&'a Arc<Node<M>>, usize)>, // (node ref, index of current child)
        total_bytes: usize, // Total bytes in the data range of the iterator.
        byte_index: isize,  // The index of the current byte relative to the data range start.
    },
    Light {
        slice: &'a [M],
        is_end: bool,
    },
}

impl<'a, M> Chunks<'a, M>
where
    M: Measurable,
{
    #[inline(always)]
    pub(crate) fn new(node: &'a Arc<Node<M>>) -> Self {
        let info = node.slice_info();
        Chunks::new_with_range_at_index(node, 0, (0, info.len as usize), (0, info.width as usize)).0
    }

    #[inline(always)]
    pub(crate) fn new_with_range(
        node: &'a Arc<Node<M>>,
        byte_index_range: (usize, usize),
        char_index_range: (usize, usize),
    ) -> Self {
        Chunks::new_with_range_at_index(
            node,
            byte_index_range.0,
            byte_index_range,
            char_index_range,
        )
        .0
    }

    /// The main workhorse function for creating new `Chunks` iterators.
    ///
    /// Creates a new `Chunks` iterator from the given node, starting the
    /// iterator at the chunk containing the `at_byte` byte index (i.e. the
    /// `next()` method will yield the chunk containing that byte).  The range
    /// of the iterator is bounded by `byte_index_range`.
    ///
    /// Both `at_byte` and `byte_index_range` are relative to the beginning of
    /// of the passed node.
    ///
    /// Passing an `at_byte` equal to the max of `byte_index_range` creates an
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

        // Special-case for empty text contents.
        println!("start: {}, end: {}", start_index, end_index);
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
        let mut byte_index = at_index as isize;
        let node_stack = {
            let mut node_stack: Vec<(&Arc<Node<M>>, usize)> = Vec::new();
            let mut node_ref = node;
            loop {
                match **node_ref {
                    Node::Leaf(ref slice) => {
                        if at_index < end_index || byte_index == 0 {
                            byte_index = info.len as isize - start_index as isize;
                        } else {
                            byte_index =
                                (info.len as isize + slice.len() as isize) - start_index as isize;
                            info = SliceInfo {
                                len: index_range.1 as u64,
                                width: width_range.1 as u64,
                            };
                            (*node_stack.last_mut().unwrap()).1 += 1;
                        }
                        break;
                    }
                    Node::Branch(ref children) => {
                        let (child_i, acc_info) = children.search_index(byte_index as usize);
                        info += acc_info;
                        node_stack.push((node_ref, child_i));
                        node_ref = &children.nodes()[child_i];
                        byte_index -= acc_info.len as isize;
                    }
                }
            }
            node_stack
        };

        // Create the iterator.
        (
            Chunks {
                iter: ChunksEnum::Full {
                    node_stack: node_stack,
                    total_bytes: end_index - start_index,
                    byte_index: byte_index,
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
        let at_byte = if at_width == width_range.1 {
            index_range.1
        } else {
            (node.get_chunk_at_width(at_width).1.len as usize).max(index_range.0)
        };

        Chunks::new_with_range_at_index(node, at_byte, index_range, width_range)
    }

    pub(crate) fn from_slice(slice: &'a [M], at_end: bool) -> Self {
        Chunks {
            iter: ChunksEnum::Light {
                slice,
                is_end: at_end,
            },
            is_reversed: false,
        }
    }

    /// Reverses the direction of the iterator in-place.
    ///
    /// In other words, swaps the behavior of [`prev()`](Chunks::prev())
    /// and [`next()`](Chunks::next()).
    #[inline]
    pub fn reverse(&mut self) {
        self.is_reversed = !self.is_reversed;
    }

    /// Same as `reverse()`, but returns itself.
    ///
    /// This is useful when chaining iterator methods:
    ///
    /// ```rust
    /// # use ropey::Rope;
    /// # let rope = Rope::from_str("Hello there\n world!\n");
    /// // Enumerate the rope's chunks in reverse, starting from the end.
    /// for (i, chunk) in rope.chunks_at_byte(rope.len_bytes()).0.reversed().enumerate() {
    ///     println!("{} {}", i, chunk);
    /// #   assert_eq!(chunk, rope.chunks().nth(rope.chunks().count() - i - 1).unwrap());
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
                        total_bytes,
                        ref mut byte_index,
                    },
                ..
            } => {
                if *byte_index <= 0 {
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

                // Get the text, sliced to the appropriate range.
                let text = node.children().nodes()[*child_i].leaf_slice();
                *byte_index -= text.len() as isize;
                let text_slice = {
                    let start_byte = if *byte_index < 0 {
                        (-*byte_index) as usize
                    } else {
                        0
                    };
                    let end_byte = text
                        .len()
                        .min((total_bytes as isize - *byte_index) as usize);
                    &text[start_byte..end_byte]
                };

                // Return the text.
                return Some(text_slice);
            }

            Chunks {
                iter:
                    ChunksEnum::Light {
                        slice: text,
                        ref mut is_end,
                    },
                ..
            } => {
                if !*is_end || text.is_empty() {
                    return None;
                } else {
                    *is_end = false;
                    return Some(text);
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
                        total_bytes,
                        ref mut byte_index,
                    },
                ..
            } => {
                if *byte_index >= total_bytes as isize {
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

                // Get the text, sliced to the appropriate range.
                let text = node.children().nodes()[*child_i].leaf_slice();
                let text_slice = {
                    let start_byte = if *byte_index < 0 {
                        (-*byte_index) as usize
                    } else {
                        0
                    };
                    let end_byte = text
                        .len()
                        .min((total_bytes as isize - *byte_index) as usize);
                    &text[start_byte..end_byte]
                };

                // Book keeping.
                *byte_index += text.len() as isize;
                *child_i += 1;

                // Return the text.
                return Some(text_slice);
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
    use crate::Rope;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Lipsum {
        Lorem,
        Ipsum,
        Dolor(usize),
        Sit,
        Amet,
        Consectur(&'static str),
        Adipiscing(bool),
    }

    impl Measurable for Lipsum {
        fn width(&self) -> usize {
            match self {
                Lipsum::Lorem => 1,
                Lipsum::Ipsum => 2,
                Lipsum::Dolor(width) => *width,
                Lipsum::Sit => 0,
                Lipsum::Amet => 0,
                Lipsum::Consectur(text) => text.len(),
                Lipsum::Adipiscing(boolean) => *boolean as usize,
            }
        }
    }

    use self::Lipsum::*;
    fn lorem_ipsum() -> Vec<Lipsum> {
        (0..1400)
            .into_iter()
            .map(|num| match num % 14 {
                0 => Lorem,
                1 => Ipsum,
                2 => Dolor(4),
                3 => Sit,
                4 => Amet,
                5 => Consectur("hello"),
                6 => Adipiscing(true),
                7 => Lorem,
                8 => Ipsum,
                9 => Dolor(8),
                10 => Sit,
                11 => Amet,
                12 => Consectur("bye"),
                13 => Adipiscing(false),
                _ => unreachable!(),
            })
            .collect()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        for (from_rope, from_vec) in rope.iter().zip(lorem_ipsum().iter().map(|test| *test)) {
            assert_eq!(from_rope, from_vec);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let mut iter = rope.iter();
        while let Some(_) = iter.next() {}

        let mut i = lorem_ipsum().len();
        while let Some(b) = iter.prev() {
            i -= 1;
            assert_eq!(b, lorem_ipsum()[i]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_03() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let mut iter = rope.iter();

        iter.next();
        iter.prev();
        assert_eq!(None, iter.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_04() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let mut iter = rope.iter();
        while let Some(_) = iter.next() {}

        iter.prev();
        iter.next();
        assert_eq!(None, iter.next());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_05() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let mut iter = rope.iter();

        assert_eq!(None, iter.prev());
        iter.next();
        iter.prev();
        assert_eq!(None, iter.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_06() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
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
        let mut iter = Iter::from_slice(&[Lorem]);

        assert_eq!(Some(Lorem), iter.next());
        assert_eq!(None, iter.next());
        assert_eq!(Some(Lorem), iter.prev());
        assert_eq!(None, iter.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_at_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let lorem_ipsum = lorem_ipsum();

        let mut iter_1 = lorem_ipsum.iter().map(|lipsum| *lipsum);
        for i in 0..(rope.len() + 1) {
            let mut iter_2 = rope.iter_at_index(i);
            assert_eq!(iter_1.next(), iter_2.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_at_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let mut bytes = rope.iter_at_index(rope.len());
        assert_eq!(bytes.next(), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_at_03() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let mut iter_1 = rope.iter_at_index(rope.len());
        let lorem_ipsum = lorem_ipsum();
        let mut iter_2 = lorem_ipsum.iter().map(|lipsum| *lipsum);

        while let Some(b) = iter_2.next_back() {
            assert_eq!(iter_1.prev(), Some(b));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn exact_size_iter_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(34..301);

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
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(34..301);

        for i in 0..=slice.len() {
            let bytes = slice.iter_at(i);
            assert_eq!(slice.len() - i, bytes.len());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn exact_size_iter_03() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(34..301);

        let mut len = 0;
        let mut iter = slice.iter_at(slice.len());

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
    fn iter_reverse_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
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
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let mut iter = rope.iter_at_index(rope.len() / 3);
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
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        let mut index = 0;
        for chunk in rope.chunks() {
            assert_eq!(chunk, &lorem_ipsum()[index..(index + chunk.len())]);
            index += chunk.len();
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_02() {
        let rope = Rope::<Lipsum>::from_slice(&[]);
        let mut iter = rope.chunks();

        assert_eq!(None, iter.next());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_03() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        let mut iter = rope.chunks();

        assert_eq!(None, iter.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_04() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        let mut chunks = Vec::new();
        let mut iter = rope.chunks();

        while let Some(text) = iter.next() {
            chunks.push(text);
        }

        while let Some(text) = iter.prev() {
            assert_eq!(text, chunks.pop().unwrap());
        }

        assert!(chunks.is_empty());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_at_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        for i in 0..rope.len() {
            let (chunk, b, c) = rope.chunk_at_index(i);
            let (mut chunks, bs, cs) = rope.chunks_at_index(i);

            assert_eq!(b, bs);
            assert_eq!(c, cs);
            assert_eq!(Some(chunk), chunks.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_at_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(34..301);

        let (mut chunks, _, _) = slice.chunks_at_index(slice.len());
        assert_eq!(chunks.next(), None);

        let (mut chunks, _, _) = slice.chunks_at_index(slice.len());
        assert_eq!(slice.chunks().last(), chunks.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_at_03() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(34..34);

        let (mut chunks, _, _) = slice.chunks_at_index(0);
        assert_eq!(chunks.next(), None);

        let (mut chunks, _, _) = slice.chunks_at_index(0);
        assert_eq!(chunks.prev(), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_reverse_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
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
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
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
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
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
        let mut iter = Chunks::from_slice(&[Dolor(5), Sit], false);

        assert_eq!(Some([Dolor(5), Sit].as_slice()), iter.next());
        assert_eq!(None, iter.next());
        iter.reverse();
        assert_eq!(Some([Dolor(5), Sit].as_slice()), iter.next());
        assert_eq!(None, iter.next());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_sliced_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        let slice_start = 34;
        let slice_end = 301;
        let slice_start_byte = rope.width_to_index(slice_start);
        let s_end_byte = rope.width_to_index(slice_end);

        let slice_1 = rope.width_slice(slice_start..slice_end);
        let slice_2 = &lorem_ipsum()[slice_start_byte..s_end_byte];

        let mut slice_1_iter = slice_1.iter();
        let mut slice_2_iter = slice_2.iter().map(|lipsum| *lipsum);

        assert_eq!(slice_1, slice_2);
        assert_eq!(slice_1.from_index(0), slice_2[0]);
        for _ in 0..(slice_2.len() + 1) {
            assert_eq!(slice_1_iter.next(), slice_2_iter.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_at_sliced_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(34..301);
        let mut bytes = slice.iter_at(slice.len());
        assert_eq!(bytes.next(), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_at_sliced_03() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        let slice_start = 34;
        let slice_end = 301;
        let slice_start_byte = rope.width_to_index(slice_start);
        let s_end_byte = rope.width_to_index(slice_end);

        let slice_1 = rope.width_slice(slice_start..slice_end);
        let slice_2 = &lorem_ipsum()[slice_start_byte..s_end_byte];

        let mut bytes_1 = slice_1.iter_at(slice_1.len());
        let mut bytes_2 = slice_2.iter().map(|lipsum| *lipsum);
        while let Some(b) = bytes_2.next_back() {
            assert_eq!(bytes_1.prev(), Some(b));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_at_sliced_reverse_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

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
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        let slice_start = 34;
        let slice_end = 301;
        let slice_start_byte = rope.width_to_index(slice_start);
        let s_end_byte = rope.width_to_index(slice_end);

        let s1 = rope.width_slice(slice_start..slice_end);
        let s2 = &lorem_ipsum()[slice_start_byte..s_end_byte];

        let mut index = 0;
        for chunk in s1.chunks() {
            assert_eq!(chunk, &s2[index..(index + chunk.len())]);
            index += chunk.len();
        }

        assert_eq!(index, s2.len());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_sliced_reverse_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

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
        let rope = Rope::<Lipsum>::from_slice(&[]);
        let rope: Vec<Lipsum> = rope.iter().collect();
        assert_eq!(&*rope, [].as_slice())
    }
}
