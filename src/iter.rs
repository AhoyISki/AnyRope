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
use crate::slice::{RSEnum, RopeSlice};
use crate::slice_utils::width_of;
use crate::tree::{Count, Node, SliceInfo};

//==========================================================

/// An iterator over a `Rope`'s bytes.
#[derive(Debug, Clone)]
pub struct Iter<'a, M>
where
    M: Measurable,
{
    chunk_iter: Chunks<'a, M>,
    cur_chunk: &'a [M],
    idx: usize,
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
            idx: 0,
            last_call_was_prev_impl: false,
            total_len: node.slice_info().len as usize,
            remaining_len: node.slice_info().len as usize,
            is_reversed: false,
        }
    }

    #[inline(always)]
    pub(crate) fn new_with_range(
        node: &'a Arc<Node<M>>,
        byte_idx_range: (usize, usize),
        width_idx_range: (usize, usize),
    ) -> Self {
        Iter::new_with_range_at(node, byte_idx_range.0, byte_idx_range, width_idx_range)
    }

    pub(crate) fn new_with_range_at(
        node: &'a Arc<Node<M>>,
        at_idx: usize,
        idx_range: (usize, usize),
        width_range: (usize, usize),
    ) -> Self {
        let (mut chunk_iter, mut chunk_start_idx, _) =
            Chunks::new_with_range_at_idx(node, at_idx, idx_range, width_range);

        let cur_chunk = if idx_range.0 == idx_range.1 {
            &[]
        } else if at_idx < idx_range.1 {
            chunk_iter.next().unwrap()
        } else {
            let chunk = chunk_iter.prev().unwrap();
            chunk_iter.next();
            chunk_start_idx -= chunk.len();
            chunk
        };

        Iter {
            chunk_iter,
            cur_chunk,
            idx: at_idx - chunk_start_idx,
            last_call_was_prev_impl: false,
            total_len: idx_range.1 - idx_range.0,
            remaining_len: idx_range.1 - at_idx,
            is_reversed: false,
        }
    }

    #[inline(always)]
    pub(crate) fn from_slice(slice: &'a [M]) -> Self {
        Iter::from_slice_at(slice, 0)
    }

    pub(crate) fn from_slice_at(slice: &'a [M], idx: usize) -> Self {
        let mut chunk_iter = Chunks::from_slice(slice, false);
        let cur_chunk = if let Some(chunk) = chunk_iter.next() {
            chunk
        } else {
            &[]
        };
        Iter {
            chunk_iter: chunk_iter,
            cur_chunk: cur_chunk,
            idx,
            last_call_was_prev_impl: false,
            total_len: slice.len(),
            remaining_len: slice.len() - idx,
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
        if self.idx == 0 {
            if let Some(chunk) = self.chunk_iter.prev() {
                self.cur_chunk = chunk;
                self.idx = self.cur_chunk.len();
            } else {
                return None;
            }
        }

        // Progress the byte counts and return the previous byte.
        let idx = self.cur_chunk[self.idx];
        self.idx -= 1;
        self.remaining_len += 1;
        return Some(idx);
    }

    #[inline]
    fn next_impl(&mut self) -> Option<M> {
        // Put us back into a "next" progression.
        if self.last_call_was_prev_impl {
            self.chunk_iter.next();
            self.last_call_was_prev_impl = false;
        }

        // Progress the chunks iterator forward if needed.
        if self.idx >= self.cur_chunk.len() {
            if let Some(chunk) = self.chunk_iter.next() {
                self.cur_chunk = chunk;
                self.idx = 0;
            } else {
                return None;
            }
        }

        // Progress the byte counts and return the next byte.
        let idx = self.cur_chunk[self.idx];
        self.idx += 1;
        self.remaining_len -= 1;
        return Some(idx);
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
        byte_idx: isize,    // The index of the current byte relative to the data range start.
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
        Chunks::new_with_range_at_idx(node, 0, (0, info.len as usize), (0, info.width as usize)).0
    }

    #[inline(always)]
    pub(crate) fn new_with_range(
        node: &'a Arc<Node<M>>,
        byte_idx_range: (usize, usize),
        char_idx_range: (usize, usize),
    ) -> Self {
        Chunks::new_with_range_at_idx(node, byte_idx_range.0, byte_idx_range, char_idx_range).0
    }

    /// The main workhorse function for creating new `Chunks` iterators.
    ///
    /// Creates a new `Chunks` iterator from the given node, starting the
    /// iterator at the chunk containing the `at_byte` byte index (i.e. the
    /// `next()` method will yield the chunk containing that byte).  The range
    /// of the iterator is bounded by `byte_idx_range`.
    ///
    /// Both `at_byte` and `byte_idx_range` are relative to the beginning of
    /// of the passed node.
    ///
    /// Passing an `at_byte` equal to the max of `byte_idx_range` creates an
    /// iterator at the end of forward iteration.
    ///
    /// Returns the iterator and the index/width of its start relative
    /// to the start of the node.
    pub(crate) fn new_with_range_at_idx(
        node: &Arc<Node<M>>,
        at_idx: usize,
        idx_range: (usize, usize),
        width_range: (usize, usize),
    ) -> (Chunks<M>, usize, usize) {
        debug_assert!(at_idx >= idx_range.0);
        debug_assert!(at_idx <= idx_range.1);

        // For convenience/readability.
        let start_idx = idx_range.0;
        let end_idx = idx_range.1;

        // Special-case for empty text contents.
        if start_idx == end_idx {
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
            let slice = &node.leaf_slice()[start_idx..end_idx];
            if at_idx == end_idx {
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
        let mut byte_idx = at_idx as isize;
        let node_stack = {
            let mut node_stack: Vec<(&Arc<Node<M>>, usize)> = Vec::new();
            let mut node_ref = node;
            loop {
                match **node_ref {
                    Node::Leaf(ref text) => {
                        if at_idx < end_idx || byte_idx == 0 {
                            byte_idx = info.len as isize - start_idx as isize;
                        } else {
                            byte_idx =
                                (info.len as isize + text.len() as isize) - start_idx as isize;
                            info = SliceInfo {
                                len: idx_range.1 as u64,
                                width: width_range.1 as u64,
                            };
                            (*node_stack.last_mut().unwrap()).1 += 1;
                        }
                        break;
                    }
                    Node::Branch(ref children) => {
                        let (child_i, acc_info) = children.search_byte_idx(byte_idx as usize);
                        info += acc_info;
                        node_stack.push((node_ref, child_i));
                        node_ref = &children.nodes()[child_i];
                        byte_idx -= acc_info.len as isize;
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
                    total_bytes: end_idx - start_idx,
                    byte_idx: byte_idx,
                },
                is_reversed: false,
            },
            (info.len as usize).max(idx_range.0),
            (info.width as usize).max(width_range.0),
        )
    }

    #[inline(always)]
    pub(crate) fn new_with_range_at_width(
        node: &'a Arc<Node<M>>,
        at_width: usize,
        idx_range: (usize, usize),
        width_range: (usize, usize),
    ) -> (Self, usize, usize) {
        let at_byte = if at_width == width_range.1 {
            idx_range.1
        } else {
            (node.get_chunk_at_width(at_width).1.len as usize).max(idx_range.0)
        };

        Chunks::new_with_range_at_idx(node, at_byte, idx_range, width_range)
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
                        ref mut byte_idx,
                    },
                ..
            } => {
                if *byte_idx <= 0 {
                    return None;
                }

                // Progress the node stack if needed.
                let mut stack_idx = node_stack.len() - 1;
                if node_stack[stack_idx].1 == 0 {
                    while node_stack[stack_idx].1 == 0 {
                        if stack_idx == 0 {
                            return None;
                        } else {
                            stack_idx -= 1;
                        }
                    }
                    node_stack[stack_idx].1 -= 1;
                    while stack_idx < (node_stack.len() - 1) {
                        let child_i = node_stack[stack_idx].1;
                        let node = &node_stack[stack_idx].0.children().nodes()[child_i];
                        node_stack[stack_idx + 1] = (node, node.child_count() - 1);
                        stack_idx += 1;
                    }
                    node_stack[stack_idx].1 += 1;
                }

                // Fetch the node and child index.
                let (node, ref mut child_i) = node_stack.last_mut().unwrap();
                *child_i -= 1;

                // Get the text, sliced to the appropriate range.
                let text = node.children().nodes()[*child_i].leaf_slice();
                *byte_idx -= text.len() as isize;
                let text_slice = {
                    let start_byte = if *byte_idx < 0 {
                        (-*byte_idx) as usize
                    } else {
                        0
                    };
                    let end_byte = text.len().min((total_bytes as isize - *byte_idx) as usize);
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
                        ref mut byte_idx,
                    },
                ..
            } => {
                if *byte_idx >= total_bytes as isize {
                    return None;
                }

                // Progress the node stack if needed.
                let mut stack_idx = node_stack.len() - 1;
                if node_stack[stack_idx].1 >= node_stack[stack_idx].0.child_count() {
                    while node_stack[stack_idx].1 >= (node_stack[stack_idx].0.child_count() - 1) {
                        if stack_idx == 0 {
                            return None;
                        } else {
                            stack_idx -= 1;
                        }
                    }
                    node_stack[stack_idx].1 += 1;
                    while stack_idx < (node_stack.len() - 1) {
                        let child_i = node_stack[stack_idx].1;
                        let node = &node_stack[stack_idx].0.children().nodes()[child_i];
                        node_stack[stack_idx + 1] = (node, 0);
                        stack_idx += 1;
                    }
                }

                // Fetch the node and child index.
                let (node, ref mut child_i) = node_stack.last_mut().unwrap();

                // Get the text, sliced to the appropriate range.
                let text = node.children().nodes()[*child_i].leaf_slice();
                let text_slice = {
                    let start_byte = if *byte_idx < 0 {
                        (-*byte_idx) as usize
                    } else {
                        0
                    };
                    let end_byte = text.len().min((total_bytes as isize - *byte_idx) as usize);
                    &text[start_byte..end_byte]
                };

                // Book keeping.
                *byte_idx += text.len() as isize;
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
    const SLICE: &[Lipsum] = &[
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
        Lorem,
        Ipsum,
        Dolor(3),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
    ];

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_01() {
        let r = Rope::from_slice(SLICE);
        for (br, bt) in r.iter().zip(SLICE.iter().map(|test| *test)) {
            assert_eq!(br, bt);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_02() {
        let r = Rope::from_slice(SLICE);
        let mut itr = r.iter();
        while let Some(_) = itr.next() {}

        let mut i = SLICE.len();
        while let Some(b) = itr.prev() {
            i -= 1;
            assert_eq!(b, SLICE[i]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_03() {
        let r = Rope::from_slice(SLICE);
        let mut itr = r.iter();

        itr.next();
        itr.prev();
        assert_eq!(None, itr.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_04() {
        let r = Rope::from_slice(SLICE);
        let mut itr = r.iter();
        while let Some(_) = itr.next() {}

        itr.prev();
        itr.next();
        assert_eq!(None, itr.next());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_05() {
        let r = Rope::from_slice(SLICE);
        let mut itr = r.iter();

        assert_eq!(None, itr.prev());
        itr.next();
        itr.prev();
        assert_eq!(None, itr.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_06() {
        let r = Rope::from_slice(SLICE);
        let mut itr = r.iter();
        while let Some(_) = itr.next() {}

        assert_eq!(None, itr.next());
        itr.prev();
        itr.next();
        assert_eq!(None, itr.next());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_07() {
        let mut itr = Iter::from_slice(&[Lorem]);

        assert_eq!(Some(0x61), itr.next());
        assert_eq!(None, itr.next());
        assert_eq!(Some(0x61), itr.prev());
        assert_eq!(None, itr.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_at_01() {
        let r = Rope::from_slice(SLICE);

        let mut bytes_1 = SLICE;
        for i in 0..(r.total_len() + 1) {
            let mut bytes_2 = r.iter_at_idx(i);
            assert_eq!(bytes_1.next(), bytes_2.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_at_02() {
        let r = Rope::from_slice(SLICE);
        let mut bytes = r.iter_at_idx(r.total_len());
        assert_eq!(bytes.next(), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_at_03() {
        let r = Rope::from_slice(SLICE);
        let mut bytes_1 = r.iter_at_idx(r.total_len());
        let mut bytes_2 = SLICE.iter().map(|lipsum| *lipsum);

        while let Some(b) = bytes_2.next_back() {
            assert_eq!(bytes_1.prev(), Some(b));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_exact_size_iter_01() {
        let r = Rope::from_slice(SLICE);
        let s = r.slice(34..301);

        let mut byte_count = s.total_len();
        let mut bytes = s.iter();

        assert_eq!(byte_count, bytes.len());

        while let Some(_) = bytes.next() {
            byte_count -= 1;
            assert_eq!(byte_count, bytes.len());
        }

        bytes.next();
        bytes.next();
        bytes.next();
        bytes.next();
        bytes.next();
        bytes.next();
        bytes.next();
        assert_eq!(bytes.len(), 0);
        assert_eq!(byte_count, 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_exact_size_iter_02() {
        let r = Rope::from_slice(SLICE);
        let s = r.slice(34..301);

        for i in 0..=s.total_len() {
            let bytes = s.iter_at(i);
            assert_eq!(s.total_len() - i, bytes.len());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_exact_size_iter_03() {
        let r = Rope::from_slice(SLICE);
        let s = r.slice(34..301);

        let mut byte_count = 0;
        let mut bytes = s.iter_at(s.total_len());

        assert_eq!(byte_count, bytes.len());

        while bytes.prev().is_some() {
            byte_count += 1;
            assert_eq!(byte_count, bytes.len());
        }

        assert_eq!(bytes.len(), s.len_idx());
        bytes.prev();
        bytes.prev();
        bytes.prev();
        bytes.prev();
        bytes.prev();
        bytes.prev();
        bytes.prev();
        assert_eq!(bytes.len(), s.len_idx());
        assert_eq!(byte_count, s.len_idx());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_reverse_01() {
        let r = Rope::from_slice(SLICE);
        let mut itr = r.iter();
        let mut stack = Vec::new();

        for _ in 0..32 {
            stack.push(itr.next().unwrap());
        }
        itr.reverse();
        for _ in 0..32 {
            assert_eq!(stack.pop(), itr.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_reverse_02() {
        let r = Rope::from_slice(SLICE);
        let mut itr = r.iter_at(r.total_len() / 3);
        let mut stack = Vec::new();

        for _ in 0..32 {
            stack.push(itr.next().unwrap());
        }
        itr.reverse();
        for _ in 0..32 {
            assert_eq!(stack.pop(), itr.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_reverse_03() {
        let r = Rope::from_slice(SLICE);
        let mut itr = r.iter_at(r.total_lent, 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_01() {
        let r = Rope::from_slice(SLICE);

        let mut idx = 0;
        for chunk in r.chunks() {
            assert_eq!(chunk, &SLICE[idx..(idx + chunk.len())]);
            idx += chunk.len();
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_02() {
        let r = Rope::from_slice("");
        let mut itr = r.chunks();

        assert_eq!(None, itr.next());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_03() {
        let r = Rope::from_slice(SLICE);

        let mut itr = r.chunks();

        assert_eq!(None, itr.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_04() {
        let r = Rope::from_slice(SLICE);

        let mut chunks = Vec::new();
        let mut itr = r.chunks();

        while let Some(text) = itr.next() {
            chunks.push(text);
        }

        while let Some(text) = itr.prev() {
            assert_eq!(text, chunks.pop().unwrap());
        }

        assert!(chunks.is_empty());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_at_byte_01() {
        let r = Rope::from_slice(SLICE);

        for i in 0..r.len_bytes() {
            let (chunk, b, c, l) = r.chunk_at_idx(i);
            let (mut chunks, bs, cs, ls) = r.chunks_at_idx(i);

            assert_eq!(b, bs);
            assert_eq!(c, cs);
            assert_eq!(l, ls);
            assert_eq!(Some(chunk), chunks.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_at_byte_02() {
        let r = Rope::from_slice(SLICE);
        let s = r.slice(34..301);

        for i in 0..(s.len_chars() + 1) {
            let (chunk, b, c, l) = s.chunk_at_byte(i);
            let (mut chunks, bs, cs, ls) = s.chunks_at(i);

            assert_eq!(b, bs);
            assert_eq!(c, cs);
            assert_eq!(l, ls);
            assert_eq!(Some(chunk), chunks.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_at_byte_03() {
        let r = Rope::from_slice(SLICE);
        let s = r.slice(34..301);

        let (mut chunks, _, _, _) = s.chunks_at(s.len_bytes());
        assert_eq!(chunks.next(), None);

        let (mut chunks, _, _, _) = s.chunks_at(s.len_bytes());
        assert_eq!(s.chunks().last(), chunks.prev());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_at_byte_04() {
        let r = Rope::from_slice(SLICE);
        let s = r.slice(34..34);

        let (mut chunks, _, _, _) = s.chunks_at(0);
        assert_eq!(chunks.next(), None);

        let (mut chunks, _, _, _) = s.chunks_at(0);
        assert_eq!(chunks.prev(), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_reverse_01() {
        let r = Rope::from_slice(SLICE);
        let mut itr = r.chunks();
        let mut stack = Vec::new();

        for _ in 0..8 {
            stack.push(itr.next().unwrap());
        }
        itr.reverse();
        for _ in 0..8 {
            assert_eq!(stack.pop().unwrap(), itr.next().unwrap());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_reverse_02() {
        let r = Rope::from_slice(SLICE);
        let mut itr = r.chunks_at_char(r.len_width() / 3).0;
        let mut stack = Vec::new();

        for _ in 0..8 {
            stack.push(itr.next().unwrap());
        }
        itr.reverse();
        for _ in 0..8 {
            assert_eq!(stack.pop().unwrap(), itr.next().unwrap());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_reverse_03() {
        let r = Rope::from_slice(SLICE);
        let mut itr = r.chunks_at_char(r.len_width() / 3).0;
        let mut stack = Vec::new();

        itr.reverse();
        for _ in 0..8 {
            stack.push(itr.next().unwrap());
        }
        itr.reverse();
        for _ in 0..8 {
            assert_eq!(stack.pop().unwrap(), itr.next().unwrap());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_reverse_04() {
        let mut itr = Chunks::from_slice("a\n", false);

        assert_eq!(Some("a\n"), itr.next());
        assert_eq!(None, itr.next());
        itr.reverse();
        assert_eq!(Some("a\n"), itr.next());
        assert_eq!(None, itr.next());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_sliced_01() {
        let r = Rope::from_slice(SLICE);

        let s_start = 34;
        let s_end = 301;
        let s_start_byte = r.width_to_idx(s_start);
        let s_end_byte = r.width_to_idx(s_end);

        let s1 = r.slice(s_start..s_end);
        let s2 = &SLICE[s_start_byte..s_end_byte];

        let mut s1_iter = s1.iter();
        let mut s2_iter = s2.iter();

        assert_eq!(s1, s2);
        assert_eq!(s1.from_idx(0), s2[0]);
        for _ in 0..(s2.len() + 1) {
            assert_eq!(s1_iter.next(), s2_iter.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_at_sliced_02() {
        let r = Rope::from_slice(SLICE);
        let s = r.slice(34..301);
        let mut bytes = s.iter_at(s.total_len());
        assert_eq!(bytes.next(), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn iter_at_sliced_03() {
        let r = Rope::from_slice(SLICE);

        let s_start = 34;
        let s_end = 301;
        let s_start_byte = r.char_to_byte(s_start);
        let s_end_byte = r.char_to_byte(s_end);

        let s1 = r.slice(s_start..s_end);
        let s2 = &SLICE[s_start_byte..s_end_byte];

        let mut bytes_1 = s1.iter_at(s1.total_len());
        let mut bytes_2 = s2.bytes();
        while let Some(b) = bytes_2.next_back() {
            assert_eq!(bytes_1.prev(), Some(b));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bytes_at_sliced_reverse_01() {
        let r = Rope::from_slice(SLICE);

        let s_start = 34;
        let s_end = 301;
        let s = r.slice(s_start..s_end);

        let mut itr = s.iter_at(s.total_len() / 3);
        let mut stack = Vec::new();
        for _ in 0..32 {
            stack.push(itr.next().unwrap());
        }
        itr.reverse();
        for _ in 0..32 {
            assert_eq!(stack.pop(), itr.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_sliced_01() {
        let r = Rope::from_slice(SLICE);

        let s_start = 34;
        let s_end = 301;
        let s_start_byte = r.char_to_byte(s_start);
        let s_end_byte = r.char_to_byte(s_end);

        let s1 = r.slice(s_start..s_end);
        let s2 = &SLICE[s_start_byte..s_end_byte];

        let mut idx = 0;
        for chunk in s1.chunks() {
            assert_eq!(chunk, &s2[idx..(idx + chunk.len())]);
            idx += chunk.len();
        }

        assert_eq!(idx, s2.len());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn chunks_sliced_reverse_01() {
        let r = Rope::from_slice(SLICE);

        let s_start = 34;
        let s_end = 301;
        let s = r.slice(s_start..s_end);

        let mut itr = s.chunks();
        let mut stack = Vec::new();
        for _ in 0..8 {
            stack.push(itr.next().unwrap());
        }
        itr.reverse();
        for _ in 0..8 {
            assert_eq!(stack.pop(), itr.next());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn empty_iter() {
        let rope = Rope::from_slice(&[]);
        let r: Vec<_> = rope.lines().collect();
        assert_eq!(&[""], &*r)
    }
}
