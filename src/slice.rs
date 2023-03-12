use std::fmt::Debug;
use std::ops::RangeBounds;
use std::sync::Arc;

use crate::iter::{Chunks, Iter};
use crate::rope::{Measurable, Rope};
use crate::slice_utils::{first_width_to_index, index_to_width, last_width_to_index, width_of};
use crate::tree::{Count, Node, SliceInfo};
use crate::{end_bound_to_num, start_bound_to_num, Error, Result};

/// An immutable view into part of a `Rope`.
///
/// Just like standard `&str` slices, `RopeSlice`s behave as if the text in
/// their range is the only text that exists.  All indexing is relative to
/// the start of their range, and all iterators and methods that return text
/// truncate that text to the range of the slice.
///
/// In other words, the behavior of a `RopeSlice` is always identical to that
/// of a full `Rope` created from the same text range.  Nothing should be
/// surprising here.
#[derive(Copy, Clone)]
pub struct RopeSlice<'a, M>(pub(crate) RSEnum<'a, M>)
where
    M: Measurable;

#[derive(Copy, Clone, Debug)]
pub(crate) enum RSEnum<'a, M>
where
    M: Measurable,
{
    Full {
        node: &'a Arc<Node<M>>,
        start_info: SliceInfo,
        end_info: SliceInfo,
    },
    Light {
        slice: &'a [M],
        count: Count,
    },
}

impl<'a, M> RopeSlice<'a, M>
where
    M: Measurable,
{
    /// Used for tests and debugging purposes.
    #[allow(dead_code)]
    pub(crate) fn is_light(&self) -> bool {
        matches!(&self.0, RSEnum::Light { .. })
    }

    pub(crate) fn new_with_range(node: &'a Arc<Node<M>>, start: usize, end: usize) -> Self {
        assert!(start <= end);
        assert!(end <= node.slice_info().width as usize);

        // Early-out shortcut for taking a slice of the full thing.
        if start == 0 && end == node.width() {
            if node.is_leaf() {
                let text = node.leaf_slice();
                return RopeSlice(RSEnum::Light {
                    slice: text,
                    count: (end - start) as Count,
                });
            } else {
                return RopeSlice(RSEnum::Full {
                    node,
                    start_info: SliceInfo { len: 0, width: 0 },
                    end_info: SliceInfo {
                        len: node.len() as Count,
                        width: node.width() as Count,
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
                    let start_index = first_width_to_index(slice, n_start);
                    let end_index =
                        start_index + last_width_to_index(&slice[start_index..], n_end - n_start);
                    return RopeSlice(RSEnum::Light {
                        slice: &slice[start_index..end_index],
                        count: (n_end - n_start) as Count,
                    });
                }

                Node::Branch(ref children) => {
                    let mut start_width = 0;
                    for (i, (info, zero_width_end)) in children.info().iter().enumerate() {
                        if n_start >= start_width && n_end <= (start_width + info.width as usize) {
                            if *zero_width_end {
                                break;
                            }
                            n_start -= start_width;
                            n_end -= start_width;
                            node = &children.nodes()[i];
                            continue 'outer;
                        }
                        start_width += info.width as usize;
                    }
                    break;
                }
            }
        }

        // Create the slice
        RopeSlice(RSEnum::Full {
            node,
            start_info: node.first_width_to_slice_info(n_start),
            end_info: node.last_width_to_slice_info(n_end),
        })
    }

    pub(crate) fn new_with_index_range(
        node: &'a Arc<Node<M>>,
        start: usize,
        end: usize,
    ) -> Result<Self> {
        assert!(start <= end);
        assert!(end <= node.slice_info().len as usize);

        // Early-out shortcut for taking a slice of the full thing.
        if start == 0 && end == node.len() {
            if node.is_leaf() {
                let text = node.leaf_slice();
                return Ok(RopeSlice(RSEnum::Light {
                    slice: text,
                    count: width_of(text) as Count,
                }));
            } else {
                return Ok(RopeSlice(RSEnum::Full {
                    node,
                    start_info: SliceInfo { len: 0, width: 0 },
                    end_info: SliceInfo {
                        len: node.len() as Count,
                        width: node.len() as Count,
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
                Node::Leaf(ref text) => {
                    let start_byte = n_start;
                    let end_byte = n_end;
                    return Ok(RopeSlice(RSEnum::Light {
                        slice: &text[start_byte..end_byte],
                        count: width_of(&text[start_byte..end_byte]) as Count,
                    }));
                }

                Node::Branch(ref children) => {
                    let mut start_byte = 0;
                    for (i, (info, _)) in children.info().iter().enumerate() {
                        if n_start >= start_byte && n_end <= (start_byte + info.len as usize) {
                            n_start -= start_byte;
                            n_end -= start_byte;
                            node = &children.nodes()[i];
                            continue 'outer;
                        }
                        start_byte += info.len as usize;
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

    /// Total number of bytes in the `RopeSlice`.
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
            RopeSlice(RSEnum::Light { slice, .. }) => slice.len(),
        }
    }

    /// Total number of chars in the `RopeSlice`.
    ///
    /// Runs in O(1) time.
    #[inline]
    pub fn width(&self) -> usize {
        match *self {
            RopeSlice(RSEnum::Full {
                end_info,
                start_info,
                ..
            }) => (end_info.width - start_info.width) as usize,
            RopeSlice(RSEnum::Light {
                count: char_count, ..
            }) => char_count as usize,
        }
    }

    //-----------------------------------------------------------------------
    // Index conversion methods

    /// Returns the char index of the given byte.
    ///
    /// Notes:
    ///
    /// - If the byte is in the middle of a multi-byte char, returns the
    ///   index of the char that the byte belongs to.
    /// - `byte_index` can be one-past-the-end, which will return one-past-the-end
    ///   char index.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_index` is out of bounds (i.e. `byte_index > len_bytes()`).
    #[inline]
    pub fn index_to_width(&self, index: usize) -> usize {
        self.try_index_to_width(index).unwrap()
    }

    /// Returns the byte index of the given char.
    ///
    /// Notes:
    ///
    /// - `char_index` can be one-past-the-end, which will return
    ///   one-past-the-end byte index.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_index` is out of bounds (i.e. `char_index > len_chars()`).
    #[inline]
    pub fn width_to_index(&self, width: usize) -> usize {
        self.try_width_to_index(width).unwrap()
    }

    //-----------------------------------------------------------------------
    // Fetch methods

    /// Returns the byte at `byte_index`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_index` is out of bounds (i.e. `byte_index >= len_bytes()`).
    #[inline]
    pub fn from_index(&self, index: usize) -> M {
        // Bounds check
        if let Some(out) = self.get_from_index(index) {
            out
        } else {
            panic!(
                "Attempt to index past end of slice: byte index {}, slice byte length {}",
                index,
                self.len()
            );
        }
    }

    /// Returns the char at `char_index`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_index` is out of bounds (i.e. `char_index >= len_chars()`).
    #[inline]
    pub fn from_width(&self, width: usize) -> M {
        if let Some(out) = self.get_from_width(width) {
            out
        } else {
            panic!(
                "Attempt to index past end of slice: char index {}, slice char length {}",
                width,
                self.width()
            );
        }
    }

    /// Returns the chunk containing the given byte index.
    ///
    /// Also returns the byte and char indices of the beginning of the chunk
    /// and the index of the line that the chunk starts on.
    ///
    /// Note: for convenience, a one-past-the-end `byte_index` returns the last
    /// chunk of the `RopeSlice`.
    ///
    /// The return value is organized as
    /// `(chunk, chunk_byte_index, chunk_char_index, chunk_line_index)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_index` is out of bounds (i.e. `byte_index > len_bytes()`).
    pub fn chunk_at_index(&self, index: usize) -> (&'a [M], usize, usize) {
        self.try_chunk_at_index(index).unwrap()
    }

    /// Returns the chunk containing the given char index.
    ///
    /// Also returns the byte and char indices of the beginning of the chunk
    /// and the index of the line that the chunk starts on.
    ///
    /// Note: for convenience, a one-past-the-end `char_index` returns the last
    /// chunk of the `RopeSlice`.
    ///
    /// The return value is organized as
    /// `(chunk, chunk_byte_index, chunk_char_index, chunk_line_index)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_index` is out of bounds (i.e. `char_index > len_chars()`).
    pub fn chunk_at_width(&self, width: usize) -> (&'a [M], usize, usize) {
        if let Some(out) = self.get_chunk_at_width(width) {
            out
        } else {
            panic!(
                "Attempt to index past end of slice: char index {}, slice char length {}",
                width,
                self.width()
            );
        }
    }

    /// Returns the entire contents of the `RopeSlice` as a `&str` if
    /// possible.
    ///
    /// This is useful for optimizing cases where the slice is only a few
    /// characters or words, and therefore has a high chance of being
    /// contiguous in memory.
    ///
    /// For large slices this method will typically fail and return `None`
    /// because large slices usually cross chunk boundaries in the rope.
    ///
    /// (Also see the `From` impl for converting to a `Cow<str>`.)
    ///
    /// Runs in O(1) time.
    #[inline]
    pub fn as_slice(&self) -> Option<&'a [M]> {
        match *self {
            RopeSlice(RSEnum::Full { .. }) => None,
            RopeSlice(RSEnum::Light { slice, .. }) => Some(slice),
        }
    }

    //-----------------------------------------------------------------------
    // Slice creation

    /// Returns a sub-slice of the `RopeSlice` in the given char index range.
    ///
    /// Uses range syntax, e.g. `2..7`, `2..`, etc.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end, or the end
    /// is out of bounds (i.e. `end > len_chars()`).
    pub fn width_slice<R>(&self, char_range: R) -> Self
    where
        R: RangeBounds<usize>,
    {
        let (start, end) = {
            let start_range = start_bound_to_num(char_range.start_bound());
            let end_range = end_bound_to_num(char_range.end_bound());

            // Early-out shortcut for taking a slice of the full thing.
            if start_range == None && end_range == None {
                return *self;
            }

            (
                start_range.unwrap_or(0),
                end_range.unwrap_or_else(|| self.width()),
            )
        };

        // Bounds check
        assert!(start <= end);
        assert!(
            end <= self.width(),
            "Attempt to slice past end of RopeSlice: slice end {}, RopeSlice length {}",
            end,
            self.width()
        );

        match *self {
            RopeSlice(RSEnum::Full {
                node, start_info, ..
            }) => RopeSlice::new_with_range(
                node,
                start_info.width as usize + start,
                start_info.width as usize + end,
            ),
            RopeSlice(RSEnum::Light { slice: text, .. }) => {
                let start_byte = first_width_to_index(text, start);
                let end_byte = last_width_to_index(text, end);
                let new_text = &text[start_byte..end_byte];
                RopeSlice(RSEnum::Light {
                    slice: new_text,
                    count: (end - start) as Count,
                })
            }
        }
    }

    /// Returns a sub-slice of the `RopeSlice` in the given byte index range.
    ///
    /// Uses range syntax, e.g. `2..7`, `2..`, etc.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The start of the range is greater than the end.
    /// - The end is out of bounds (i.e. `end > len_bytes()`).
    /// - The range doesn't align with char boundaries.
    pub fn index_slice<R>(&self, byte_range: R) -> Self
    where
        R: RangeBounds<usize>,
    {
        match self.get_slice_impl(byte_range) {
            Ok(s) => return s,
            Err(e) => panic!("slice(): {}", e),
        }
    }

    //-----------------------------------------------------------------------
    // Iterator methods

    /// Creates an iterator over the bytes of the `RopeSlice`.
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
                (start_info.width as usize, end_info.width as usize),
            ),
            RopeSlice(RSEnum::Light { slice: text, .. }) => Iter::from_slice(text),
        }
    }

    /// Creates an iterator over the bytes of the `RopeSlice`, starting at
    /// byte `byte_index`.
    ///
    /// If `byte_index == len_bytes()` then an iterator at the end of the
    /// `RopeSlice` is created (i.e. `next()` will return `None`).
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_index` is out of bounds (i.e. `byte_index > len_bytes()`).
    #[inline]
    pub fn iter_at(&self, byte_index: usize) -> Iter<'a, M> {
        if let Some(out) = self.get_iter_at(byte_index) {
            out
        } else {
            panic!(
                "Attempt to index past end of RopeSlice: byte index {}, RopeSlice byte length {}",
                byte_index,
                self.len()
            );
        }
    }

    /// Creates an iterator over t
    /// Creates an iterator over the chunks of the `RopeSlice`.
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
                (start_info.width as usize, end_info.width as usize),
            ),
            RopeSlice(RSEnum::Light { slice: text, .. }) => Chunks::from_slice(text, false),
        }
    }

    /// Creates an iterator over the chunks of the `RopeSlice`, with the
    /// iterator starting at the byte containing `byte_index`.
    ///
    /// Also returns the byte and char indices of the beginning of the first
    /// chunk to be yielded, and the index of the line that chunk starts on.
    ///
    /// If `byte_index == len_bytes()` an iterator at the end of the `RopeSlice`
    /// (yielding `None` on a call to `next()`) is created.
    ///
    /// The return value is organized as
    /// `(iterator, chunk_byte_index, chunk_char_index, chunk_line_index)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_index` is out of bounds (i.e. `byte_index > len_bytes()`).
    #[inline]
    pub fn chunks_at_index(&self, index: usize) -> (Chunks<'a, M>, usize, usize) {
        if let Some(out) = self.get_chunks_at_index(index) {
            out
        } else {
            panic!(
                "Attempt to index past end of RopeSlice: byte index {}, RopeSlice byte length {}",
                index,
                self.len()
            );
        }
    }

    /// Creates an iterator over the chunks of the `RopeSlice`, with the
    /// iterator starting on the chunk containing `char_index`.
    ///
    /// Also returns the byte and char indices of the beginning of the first
    /// chunk to be yielded, and the index of the line that chunk starts on.
    ///
    /// If `char_index == len_chars()` an iterator at the end of the `RopeSlice`
    /// (yielding `None` on a call to `next()`) is created.
    ///
    /// The return value is organized as
    /// `(iterator, chunk_byte_index, chunk_char_index, chunk_line_index)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_index` is out of bounds (i.e. `char_index > len_chars()`).
    #[inline]
    pub fn chunks_at_width(&self, width_index: usize) -> (Chunks<'a, M>, usize, usize) {
        if let Some(out) = self.get_chunks_at_width(width_index) {
            out
        } else {
            panic!(
                "Attempt to index past end of RopeSlice: char index {}, RopeSlice char length {}",
                width_index,
                self.width()
            );
        }
    }
}

/// # Non-Panicking
///
/// The methods in this impl block provide non-panicking versions of
/// `RopeSlice`'s panicking methods.  They return either `Option::None` or
/// `Result::Err()` when their panicking counterparts would have panicked.
impl<'a, M> RopeSlice<'a, M>
where
    M: Measurable,
{
    /// Non-panicking version of [`byte_to_char()`](RopeSlice::byte_to_char).
    #[inline]
    pub fn try_index_to_width(&self, byte_index: usize) -> Result<usize> {
        // Bounds check
        if byte_index <= self.len() {
            let (chunk, b, c) = self.chunk_at_index(byte_index);
            Ok(c + index_to_width(chunk, byte_index - b))
        } else {
            Err(Error::IndexOutOfBounds(byte_index, self.len()))
        }
    }

    /// Non-panicking version of [`char_to_byte()`](RopeSlice::char_to_byte).
    #[inline]
    pub fn try_width_to_index(&self, width: usize) -> Result<usize> {
        // Bounds check
        if width <= self.width() {
            let (chunk, b, c) = self.chunk_at_width(width);
            Ok(b + first_width_to_index(chunk, width - c))
        } else {
            Err(Error::WidthOutOfBounds(width, self.width()))
        }
    }

    /// Non-panicking version of [`get_byte()`](RopeSlice::get_byte).
    #[inline]
    pub fn get_from_index(&self, index: usize) -> Option<M> {
        // Bounds check
        if index < self.len() {
            let (chunk, chunk_byte_index, _) = self.chunk_at_index(index);
            let chunk_rel_byte_index = index - chunk_byte_index;
            Some(chunk[chunk_rel_byte_index])
        } else {
            None
        }
    }

    /// Non-panicking version of [`char()`](RopeSlice::char).
    #[inline]
    pub fn get_from_width(&self, width: usize) -> Option<M> {
        // Bounds check
        if width < self.width() {
            let (chunk, _, chunk_char_index) = self.chunk_at_width(width);
            let index = first_width_to_index(chunk, width - chunk_char_index);
            Some(chunk[index])
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunk_at_byte()`](RopeSlice::chunk_at_byte).
    pub fn try_chunk_at_index(&self, index: usize) -> Result<(&'a [M], usize, usize)> {
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

                    // Calculate clipped start/end byte indices within the chunk.
                    let chunk_start_byte_index =
                        start_info.len.saturating_sub(chunk_start_info.len);
                    let chunk_end_byte_index =
                        (chunk.len() as Count).min(end_info.len - chunk_start_info.len);

                    // Return the clipped chunk and byte offset.
                    Ok((
                        &chunk[chunk_start_byte_index as usize..chunk_end_byte_index as usize],
                        chunk_start_info.len.saturating_sub(start_info.len) as usize,
                        chunk_start_info.width.saturating_sub(start_info.width) as usize,
                    ))
                }
                RopeSlice(RSEnum::Light { slice: text, .. }) => Ok((text, 0, 0)),
            }
        } else {
            Err(Error::IndexOutOfBounds(index, self.len()))
        }
    }

    /// Non-panicking version of [`chunk_at_char()`](RopeSlice::chunk_at_char).
    pub fn get_chunk_at_width(&self, width: usize) -> Option<(&'a [M], usize, usize)> {
        // Bounds check
        if width <= self.width() {
            match *self {
                RopeSlice(RSEnum::Full {
                    node,
                    start_info,
                    end_info,
                }) => {
                    // Get the chunk.
                    let (chunk, chunk_start_info) =
                        node.get_first_chunk_at_width(width + start_info.width as usize);

                    // Calculate clipped start/end byte indices within the chunk.
                    let chunk_start_byte_index =
                        start_info.len.saturating_sub(chunk_start_info.len);
                    let chunk_end_byte_index =
                        (chunk.len() as Count).min(end_info.len - chunk_start_info.len);

                    // Return the clipped chunk and byte offset.
                    Some((
                        &chunk[chunk_start_byte_index as usize..chunk_end_byte_index as usize],
                        chunk_start_info.len.saturating_sub(start_info.len) as usize,
                        chunk_start_info.width.saturating_sub(start_info.width) as usize,
                    ))
                }
                RopeSlice(RSEnum::Light { slice: text, .. }) => Some((text, 0, 0)),
            }
        } else {
            None
        }
    }

    /// Non-panicking version of [`slice()`](RopeSlice::slice).
    pub fn get_width_slice<R>(&self, width_range: R) -> Option<RopeSlice<'a, M>>
    where
        R: RangeBounds<usize>,
    {
        let (start, end) = {
            let start_range = start_bound_to_num(width_range.start_bound());
            let end_range = end_bound_to_num(width_range.end_bound());

            // Early-out shortcut for taking a slice of the full thing.
            if start_range == None && end_range == None {
                return Some(*self);
            }

            (
                start_range.unwrap_or(0),
                end_range.unwrap_or_else(|| self.width()),
            )
        };

        // Bounds check
        if start <= end && end <= self.width() {
            match *self {
                RopeSlice(RSEnum::Full {
                    node, start_info, ..
                }) => Some(RopeSlice::new_with_range(
                    node,
                    start_info.width as usize + start,
                    start_info.width as usize + end,
                )),
                RopeSlice(RSEnum::Light { slice: text, .. }) => {
                    let start_byte = first_width_to_index(text, start);
                    let end_byte = first_width_to_index(text, end);
                    let new_text = &text[start_byte..end_byte];
                    Some(RopeSlice(RSEnum::Light {
                        slice: new_text,
                        count: (end - start) as Count,
                    }))
                }
            }
        } else {
            None
        }
    }

    /// Non-panicking version of [`slice()`](RopeSlice::slice).
    pub fn get_index_slice<R>(&self, byte_range: R) -> Option<RopeSlice<'a, M>>
    where
        R: RangeBounds<usize>,
    {
        self.get_slice_impl(byte_range).ok()
    }

    pub(crate) fn get_slice_impl<R>(&self, byte_range: R) -> Result<RopeSlice<'a, M>>
    where
        R: RangeBounds<usize>,
    {
        let start_range = start_bound_to_num(byte_range.start_bound());
        let end_range = end_bound_to_num(byte_range.end_bound());

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
            RopeSlice(RSEnum::Light { slice: text, .. }) => {
                let new_text = &text[start..end];
                Ok(RopeSlice(RSEnum::Light {
                    slice: new_text,
                    count: width_of(new_text) as Count,
                }))
            }
        }
    }

    /// Non-panicking version of [`bytes_at()`](RopeSlice::bytes_at).
    #[inline]
    pub fn get_iter_at(&self, index: usize) -> Option<Iter<'a, M>> {
        // Bounds check
        if index <= self.len() {
            match *self {
                RopeSlice(RSEnum::Full {
                    node,
                    start_info,
                    end_info,
                }) => Some(Iter::new_with_range_at(
                    node,
                    start_info.len as usize + index,
                    (start_info.len as usize, end_info.len as usize),
                    (start_info.width as usize, end_info.width as usize),
                )),
                RopeSlice(RSEnum::Light { slice: text, .. }) => {
                    Some(Iter::from_slice_at(text, index))
                }
            }
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunks_at_byte()`](RopeSlice::chunks_at_byte).
    #[inline]
    pub fn get_chunks_at_index(&self, byte_index: usize) -> Option<(Chunks<'a, M>, usize, usize)> {
        // Bounds check
        if byte_index <= self.len() {
            match *self {
                RopeSlice(RSEnum::Full {
                    node,
                    start_info,
                    end_info,
                }) => {
                    let (chunks, chunk_byte_index, chunk_char_index) =
                        Chunks::new_with_range_at_index(
                            node,
                            byte_index + start_info.len as usize,
                            (start_info.len as usize, end_info.len as usize),
                            (start_info.width as usize, end_info.width as usize),
                        );

                    Some((
                        chunks,
                        chunk_byte_index.saturating_sub(start_info.len as usize),
                        chunk_char_index.saturating_sub(start_info.width as usize),
                    ))
                }
                RopeSlice(RSEnum::Light {
                    slice: text,
                    count: char_count,
                    ..
                }) => {
                    let chunks = Chunks::from_slice(text, byte_index == text.len());

                    if byte_index == text.len() {
                        Some((chunks, text.len(), char_count as usize))
                    } else {
                        Some((chunks, 0, 0))
                    }
                }
            }
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunks_at_char()`](RopeSlice::chunks_at_char).
    #[inline]
    pub fn get_chunks_at_width(&self, width: usize) -> Option<(Chunks<'a, M>, usize, usize)> {
        // Bounds check
        if width <= self.width() {
            match *self {
                RopeSlice(RSEnum::Full {
                    node,
                    start_info,
                    end_info,
                }) => {
                    let (chunks, chunk_byte_index, chunk_char_index) =
                        Chunks::new_with_range_at_width(
                            node,
                            width + start_info.width as usize,
                            (start_info.len as usize, end_info.len as usize),
                            (start_info.width as usize, end_info.width as usize),
                        );

                    Some((
                        chunks,
                        chunk_byte_index.saturating_sub(start_info.len as usize),
                        chunk_char_index.saturating_sub(start_info.width as usize),
                    ))
                }
                RopeSlice(RSEnum::Light { slice, count, .. }) => {
                    let chunks = Chunks::from_slice(slice, width == count as usize);

                    if width == count as usize {
                        Some((chunks, slice.len(), count as usize))
                    } else {
                        Some((chunks, 0, 0))
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

/// Creates a `RopeSlice` directly from a string slice.
///
/// The useful applications of this are actually somewhat narrow.  It is
/// intended primarily as an aid when implementing additional functionality
/// on top of Ropey, where you may already have access to a rope chunk and
/// want to directly create a `RopeSlice` from it, avoiding the overhead of
/// going through the slicing APIs.
///
/// Although it is possible to use this to create `RopeSlice`s from
/// arbitrary strings, doing so is not especially useful.  For example,
/// `Rope`s and `RopeSlice`s can already be directly compared for
/// equality with strings and string slices.
///
/// Runs in O(N) time, where N is the length of the string slice.
impl<'a, M> From<&'a [M]> for RopeSlice<'a, M>
where
    M: Measurable,
{
    #[inline]
    fn from(slice: &'a [M]) -> Self {
        RopeSlice(RSEnum::Light {
            slice,
            count: width_of(slice) as Count,
        })
    }
}

impl<'a, M> From<RopeSlice<'a, M>> for Vec<M>
where
    M: Measurable,
{
    #[inline]
    fn from(s: RopeSlice<'a, M>) -> Self {
        let mut vec = Vec::with_capacity(s.len());
        vec.extend(
            s.chunks()
                .map(|chunk| chunk.iter())
                .flatten()
                .map(|measurable| *measurable),
        );
        vec
    }
}

/// Attempts to borrow the contents of the slice, but will convert to an
/// owned string if the contents is not contiguous in memory.
///
/// Runs in best case O(1), worst case O(N).
impl<'a, M> From<RopeSlice<'a, M>> for std::borrow::Cow<'a, [M]>
where
    M: Measurable,
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
    M: Measurable + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_list().entries(self.chunks()).finish()
    }
}

impl<'a, M> std::fmt::Display for RopeSlice<'a, M>
where
    M: Measurable + Debug,
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_list().entries(self.chunks()).finish()
    }
}

impl<'a, M> std::cmp::Eq for RopeSlice<'a, M> where M: Measurable + Eq {}

impl<'a, 'b, M> std::cmp::PartialEq<RopeSlice<'b, M>> for RopeSlice<'a, M>
where
    M: Measurable + PartialEq,
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
            RopeSlice(RSEnum::Light { slice: text, .. }) => {
                return text == *other;
            }
        }
    }
}

impl<'a, 'b, M> std::cmp::PartialEq<RopeSlice<'a, M>> for &'b [M]
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &RopeSlice<'a, M>) -> bool {
        other == self
    }
}

impl<'a, M> std::cmp::PartialEq<[M]> for RopeSlice<'a, M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &[M]) -> bool {
        std::cmp::PartialEq::<&[M]>::eq(self, &other)
    }
}

impl<'a, M> std::cmp::PartialEq<RopeSlice<'a, M>> for [M]
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &RopeSlice<'a, M>) -> bool {
        std::cmp::PartialEq::<&[M]>::eq(other, &self)
    }
}

impl<'a, M> std::cmp::PartialEq<Vec<M>> for RopeSlice<'a, M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Vec<M>) -> bool {
        self == other.as_slice()
    }
}

impl<'a, M> std::cmp::PartialEq<RopeSlice<'a, M>> for Vec<M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &RopeSlice<'a, M>) -> bool {
        self.as_slice() == other
    }
}

impl<'a, 'b, M> std::cmp::PartialEq<std::borrow::Cow<'b, [M]>> for RopeSlice<'a, M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &std::borrow::Cow<'b, [M]>) -> bool {
        *self == **other
    }
}

impl<'a, 'b, M> std::cmp::PartialEq<RopeSlice<'a, M>> for std::borrow::Cow<'b, [M]>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &RopeSlice<'a, M>) -> bool {
        **self == *other
    }
}

impl<'a, M> std::cmp::PartialEq<Rope<M>> for RopeSlice<'a, M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        *self == other.width_slice(..)
    }
}

impl<'a, M> std::cmp::PartialEq<RopeSlice<'a, M>> for Rope<M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &RopeSlice<'a, M>) -> bool {
        self.width_slice(..) == *other
    }
}

impl<'a, M> std::cmp::Ord for RopeSlice<'a, M>
where
    M: Measurable + Ord,
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
        rope::Lipsum::{self, *},
        slice_utils::{first_width_to_index, index_to_width},
        Rope,
    };

    /// 70 elements, total width of 135.
    fn lorem_ipsum() -> Vec<Lipsum> {
        (0..70)
            .into_iter()
            .map(|num| match num % 14 {
                0 | 7 => Lorem,
                1 | 8 => Ipsum,
                2 => Dolor(4),
                3 | 10 => Sit,
                4 | 11 => Amet,
                5 => Consectur("hello"),
                6 => Adipiscing(true),
                9 => Dolor(8),
                12 => Consectur("bye"),
                13 => Adipiscing(false),
                _ => unreachable!(),
            })
            .collect()
    }

    #[test]
    fn len_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(7..30);
        assert_eq!(slice.len(), 13);
    }

    #[test]
    fn len_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(43..43);
        assert_eq!(slice.len(), 0);
    }

    #[test]
    fn width_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(7..30);
        assert_eq!(slice.width(), 23);
    }

    #[test]
    fn width_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(43..43);
        assert_eq!(slice.width(), 0);
    }

    #[test]
    fn width_03() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(6..30);
        assert_eq!(slice.width(), 24);
    }

    #[test]
    fn index_to_width_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(88..);

        assert_eq!(slice.index_to_width(0), 0); // Sit: 0
        assert_eq!(slice.index_to_width(1), 0); // Amet: 0
        assert_eq!(slice.index_to_width(2), 0); // Consectur("hello"): 5
        assert_eq!(slice.index_to_width(3), 5); // Adipiscing(true): 1
        assert_eq!(slice.index_to_width(4), 6); // Lorem: 1
        assert_eq!(slice.index_to_width(5), 7); // Ipsum: 2
        assert_eq!(slice.index_to_width(6), 9); // Dolor(8): 8
        assert_eq!(slice.index_to_width(7), 17); // Sit: 0
        assert_eq!(slice.index_to_width(8), 17); // Amet: 0
        assert_eq!(slice.index_to_width(9), 17); // Consectur("bye"): 3
        assert_eq!(slice.index_to_width(10), 20); // Adipiscing(false): 0
        assert_eq!(slice.index_to_width(11), 20); // Sit: 1
        assert_eq!(slice.index_to_width(12), 21); // Amet: 2
        assert_eq!(slice.index_to_width(13), 23); // Dolor(4): 4
    }

    #[test]
    fn width_to_index_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(88..135);

        // NOTE: For some elements, it may seem like the amount of "widths"
        // that correspond to their index may not match their actual width.
        // For example, `Consectur("bye")` only lasts for 2 widths, even
        // though its width is 3.
        // This is because there are 0 width elements preceding it, and the
        // width in `width_to_index()` merely corresponds to the end of an
        // element, and has no relation to the referred element's width.
        assert_eq!(slice.width_to_index(0), 0); // Sit: 0
        assert_eq!(slice.width_to_index(1), 2); // Amet: 0; Consectur("hello"): 5
        assert_eq!(slice.width_to_index(4), 2); // Consectur("hello"): 5
        assert_eq!(slice.width_to_index(5), 3); // Adipiscing(true): 1
        assert_eq!(slice.width_to_index(6), 4); // Lorem: 1
        assert_eq!(slice.width_to_index(7), 5); // Ipsum: 2
        assert_eq!(slice.width_to_index(8), 5); // Ipsum: 2
        assert_eq!(slice.width_to_index(9), 6); // Dolor(8): 8
        assert_eq!(slice.width_to_index(16), 6); // Dolor(8): 8
        assert_eq!(slice.width_to_index(17), 7); // Sit: 0
        assert_eq!(slice.width_to_index(18), 9); // Amet:0; Consectur("bye"): 3
        assert_eq!(slice.width_to_index(19), 9); // Consectur("bye"): 3
        assert_eq!(slice.width_to_index(20), 10); // Adipiscing(false): 0
        assert_eq!(slice.width_to_index(21), 12); // Lorem: 1
        assert_eq!(slice.width_to_index(22), 12); // Ipsum: 2
        assert_eq!(slice.width_to_index(23), 13); // Ipsum: 2
        assert_eq!(slice.width_to_index(24), 13); // Dolor(4): 4
    }

    #[test]
    fn from_index_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(34..100);

        assert_eq!(slice.from_index(0), Sit);
        assert_eq!(slice.from_index(10), Adipiscing(false));

        assert_eq!(slice.from_index(slice.len() - 3), Lorem);
        assert_eq!(slice.from_index(slice.len() - 2), Ipsum);
        assert_eq!(slice.from_index(slice.len() - 1), Dolor(8));
    }

    #[test]
    #[should_panic]
    fn from_index_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(34..100);
        slice.from_index(slice.len());
    }

    #[test]
    #[should_panic]
    fn from_index_03() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(42..42);
        slice.from_index(0);
    }

    #[test]
    fn from_width_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(34..100);

        assert_eq!(slice.from_width(0), Sit); // Sit: 0
        assert_eq!(slice.from_width(3), Consectur("hello")); // Amet: 0; Consectur("hello"): 5
        assert_eq!(slice.from_width(5), Adipiscing(true)); // Adipiscing(true): 1
        assert_eq!(slice.from_width(6), Lorem); // Lorem: 1
        assert_eq!(slice.from_width(10), Dolor(8)); // Ipsum: 2; Dolor(8): 8
        assert_eq!(slice.from_width(65), Dolor(8)); // ...; Dolor(8): 8
    }

    #[test]
    #[should_panic]
    fn from_width_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(34..100);
        slice.from_width(66);
    }

    #[test]
    #[should_panic]
    fn from_width_03() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(43..43);
        slice.from_width(0);
    }

    #[test]
    fn chunk_at_index_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice_1 = rope.width_slice(34..135);
        let slice_2 = &lorem_ipsum()[17..70];
        // "'slice a fine day, isn't it?\nAren't you glad \
        //  we're alive?\nこんにちは、みん"

        let mut total = slice_2;
        let mut prev_chunk = [].as_slice();
        for i in 0..slice_1.len() {
            let (chunk, index, width) = slice_1.chunk_at_index(i);
            assert_eq!(width, index_to_width(slice_2, index));
            if chunk != prev_chunk {
                assert_eq!(chunk, &total[..chunk.len()]);
                total = &total[chunk.len()..];
                prev_chunk = chunk;
            }

            let lipsum_1 = slice_2.iter().nth(i).unwrap();
            let lipsum_2 = {
                let i2 = i - index;
                chunk.iter().nth(i2).unwrap()
            };
            assert_eq!(lipsum_1, lipsum_2);
        }

        assert_eq!(total.len(), 0);
    }

    #[test]
    fn chunk_at_width_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice_1 = rope.width_slice(34..96);
        let slice_2 = &lorem_ipsum()[17..51];

        let mut total = slice_2;
        let mut prev_chunk = [].as_slice();
        for i in 0..slice_1.width() {
            let (chunk, _, width) = slice_1.chunk_at_width(i);
            if chunk != prev_chunk {
                assert_eq!(chunk, &total[..chunk.len()]);
                total = &total[chunk.len()..];
                prev_chunk = chunk;
            }

            let lipsum_1 = {
                let index_1 = first_width_to_index(slice_2, i);
                slice_2.iter().nth(index_1).unwrap()
            };
            let lipsum_2 = {
                let index_2 = first_width_to_index(chunk, i - width);
                chunk.iter().nth(index_2).unwrap()
            };
            assert_eq!(lipsum_1, lipsum_2);
        }
        assert_eq!(total.len(), 0);
    }

    #[test]
    fn slice_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice_1 = rope.width_slice(..);

        let slice_2 = slice_1.width_slice(..);

        assert_eq!(lorem_ipsum(), slice_2);
    }

    #[test]
    fn slice_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice_1 = rope.width_slice(5..43);

        let slice_2 = slice_1.width_slice(3..25);

        assert_eq!(&lorem_ipsum()[5..16], slice_2);
    }

    #[test]
    fn slice_03() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice_1 = rope.width_slice(31..97);

        let slice_2 = slice_1.width_slice(7..64);

        assert_eq!(&lorem_ipsum()[19..50], slice_2);
    }

    #[test]
    fn slice_04() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice_1 = rope.width_slice(5..43);

		// A slice in the middle of a non zero width element should return only that element.
        let slice_2 = slice_1.width_slice(24..24);

        assert_eq!(slice_2, [Ipsum].as_slice());
    }

    #[test]
    #[should_panic]
    fn slice_05() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(5..43);

        #[allow(clippy::reversed_empty_ranges)]
        slice.width_slice(21..20); // Wrong ordering on purpose.
    }

    #[test]
    #[should_panic]
    fn slice_07() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(5..43);

        slice.width_slice(37..39);
    }

    #[test]
    fn eq_slice_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(..);

        assert_eq!(slice, lorem_ipsum());
        assert_eq!(lorem_ipsum(), slice);
    }

    #[test]
    fn eq_slice_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(0..20);

        assert_ne!(slice, lorem_ipsum());
        assert_ne!(lorem_ipsum(), slice);
    }

    #[test]
    fn eq_str_03() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());
        rope.remove(20..21);
        rope.insert_slice(20, &[Consectur("hello")]);
        let slice = rope.width_slice(..);

        assert_ne!(slice, lorem_ipsum());
        assert_ne!(lorem_ipsum(), slice);
    }

    #[test]
    fn eq_str_04() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(..);
        let slice: Vec<Lipsum> = lorem_ipsum().into();

        assert_eq!(slice, slice);
        assert_eq!(slice, slice);
    }

    #[test]
    fn eq_rope_slice_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(43..43);

        assert_eq!(slice, slice);
    }

    #[test]
    fn eq_rope_slice_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice_1 = rope.width_slice(43..97);
        let slice_2 = rope.width_slice(43..97);

        assert_eq!(slice_1, slice_2);
    }

    #[test]
    fn eq_rope_slice_03() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice_1 = rope.width_slice(43..43);
        let slice_2 = rope.width_slice(43..45);

        assert_ne!(slice_1, slice_2);
    }

    #[test]
    fn eq_rope_slice_04() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice_1 = rope.width_slice(43..45);
        let slice_2 = rope.width_slice(43..43);

        assert_ne!(slice_1, slice_2);
    }

    #[test]
    fn eq_rope_slice_05() {
        let rope: Rope<Lipsum> = Rope::from_slice([].as_slice());
        let slice = rope.width_slice(0..0);

        assert_eq!(slice, slice);
    }

    #[test]
    fn cmp_rope_slice_01() {
        let rope_1 = Rope::from_slice(lorem_ipsum().as_slice());
        let rope_2 = Rope::from_slice(lorem_ipsum().as_slice());
        let slice_1 = rope_1.width_slice(..);
        let slice_2 = rope_2.width_slice(..);

        assert_eq!(slice_1.cmp(&slice_2), std::cmp::Ordering::Equal);
        assert_eq!(
            slice_1.width_slice(..24).cmp(&slice_2),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            slice_1.cmp(&slice_2.width_slice(..24)),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn cmp_rope_slice_02() {
        let rope_1 = Rope::from_slice(&[Dolor(3), Amet, Consectur("hi"), Adipiscing(true)]);
        let rope_2 = Rope::from_slice(&[Dolor(3), Sit, Consectur("hi"), Adipiscing(true)]);
        let slice_1 = rope_1.width_slice(..);
        let slice_2 = rope_2.width_slice(..);

        assert_eq!(slice_1.cmp(&slice_2), std::cmp::Ordering::Greater);
        assert_eq!(slice_2.cmp(&slice_1), std::cmp::Ordering::Less);
    }

    #[test]
    fn to_vec_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(..);
        let vec: Vec<Lipsum> = slice.into();

        assert_eq!(rope, vec);
        assert_eq!(vec, slice);
    }

    #[test]
    fn to_vec_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(0..24);
        let vec: Vec<Lipsum> = slice.into();

        assert_eq!(slice, vec);
    }

    #[test]
    fn to_vec_03() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(13..89);
        let vec: Vec<Lipsum> = slice.into();

        assert_eq!(slice, vec);
    }

    #[test]
    fn to_vec_04() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(13..41);
        let vec: Vec<Lipsum> = slice.into();

        assert_eq!(slice, vec);
    }

    #[test]
    fn to_cow_01() {
        use std::borrow::Cow;
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(13..83);
        let cow: Cow<[Lipsum]> = slice.into();

        assert_eq!(slice, cow);
    }

    #[test]
    fn to_cow_02() {
        use std::borrow::Cow;
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope.width_slice(13..14);
        let cow: Cow<[Lipsum]> = rope.width_slice(13..14).into();

        // Make sure it's borrowed.
        if let Cow::Owned(_) = cow {
            panic!("Small Cow conversions should result in a borrow.");
        }

        assert_eq!(slice, cow);
    }
}
