use std::fmt::Debug;
use std::ops::RangeBounds;
use std::sync::Arc;

use crate::iter::{Chunks, Iter};
use crate::rope::{Measurable, Rope};
use crate::slice_utils::{idx_to_width, is_width_boundary, width_of, width_to_idx};
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
        if start == 0 && end == node.count() {
            if node.is_leaf() {
                let text = node.leaf_slice();
                return RopeSlice(RSEnum::Light {
                    slice: text,
                    count: (end - start) as Count,
                });
            } else {
                return RopeSlice(RSEnum::Full {
                    node: node,
                    start_info: SliceInfo { len: 0, width: 0 },
                    end_info: SliceInfo {
                        len: node.count() as Count,
                        width: node.count() as Count,
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
                Node::Leaf(ref text) => {
                    let start_idx = width_to_idx(text, n_start);
                    let end_idx = start_idx + width_to_idx(&text[start_idx..], n_end - n_start);
                    return RopeSlice(RSEnum::Light {
                        slice: &text[start_idx..end_idx],
                        count: (n_end - n_start) as Count,
                    });
                }

                Node::Branch(ref children) => {
                    let mut start_width = 0;
                    for (i, inf) in children.info().iter().enumerate() {
                        if n_start >= start_width && n_end <= (start_width + inf.width as usize) {
                            n_start -= start_width;
                            n_end -= start_width;
                            node = &children.nodes()[i];
                            continue 'outer;
                        }
                        start_width += inf.width as usize;
                    }
                    break;
                }
            }
        }

        // Create the slice
        RopeSlice(RSEnum::Full {
            node: node,
            start_info: node.width_to_slice_info(n_start),
            end_info: node.width_to_slice_info(n_end),
        })
    }

    pub(crate) fn new_with_idx_range(
        node: &'a Arc<Node<M>>,
        start: usize,
        end: usize,
    ) -> Result<Self> {
        assert!(start <= end);
        assert!(end <= node.slice_info().len as usize);

        // Early-out shortcut for taking a slice of the full thing.
        if start == 0 && end == node.count() {
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
                        len: node.count() as Count,
                        width: node.count() as Count,
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
                    for (i, inf) in children.info().iter().enumerate() {
                        if n_start >= start_byte && n_end <= (start_byte + inf.len as usize) {
                            n_start -= start_byte;
                            n_end -= start_byte;
                            node = &children.nodes()[i];
                            continue 'outer;
                        }
                        start_byte += inf.len as usize;
                    }
                    break;
                }
            }
        }

        // Create the slice
        Ok(RopeSlice(RSEnum::Full {
            node,
            start_info: node.idx_to_slice_info(n_start),
            end_info: node.idx_to_slice_info(n_end),
        }))
    }

    //-----------------------------------------------------------------------
    // Informational methods

    /// Total number of bytes in the `RopeSlice`.
    ///
    /// Runs in O(1) time.
    #[inline]
    pub fn total_len(&self) -> usize {
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
    /// - `byte_idx` can be one-past-the-end, which will return one-past-the-end
    ///   char index.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_idx` is out of bounds (i.e. `byte_idx > len_bytes()`).
    #[inline]
    pub fn idx_to_width(&self, idx: usize) -> usize {
        self.try_idx_to_width(idx).unwrap()
    }

    /// Returns the byte index of the given char.
    ///
    /// Notes:
    ///
    /// - `char_idx` can be one-past-the-end, which will return
    ///   one-past-the-end byte index.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_idx` is out of bounds (i.e. `char_idx > len_chars()`).
    #[inline]
    pub fn width_to_idx(&self, width: usize) -> usize {
        self.try_width_to_idx(width).unwrap()
    }

    //-----------------------------------------------------------------------
    // Fetch methods

    /// Returns the byte at `byte_idx`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_idx` is out of bounds (i.e. `byte_idx >= len_bytes()`).
    #[inline]
    pub fn from_idx(&self, idx: usize) -> &M {
        // Bounds check
        if let Some(out) = self.get_from_idx(idx) {
            out
        } else {
            panic!(
                "Attempt to index past end of slice: byte index {}, slice byte length {}",
                idx,
                self.total_len()
            );
        }
    }

    /// Returns the char at `char_idx`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_idx` is out of bounds (i.e. `char_idx >= len_chars()`).
    #[inline]
    pub fn from_width(&self, width: usize) -> &M {
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
    /// Note: for convenience, a one-past-the-end `byte_idx` returns the last
    /// chunk of the `RopeSlice`.
    ///
    /// The return value is organized as
    /// `(chunk, chunk_byte_idx, chunk_char_idx, chunk_line_idx)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_idx` is out of bounds (i.e. `byte_idx > len_bytes()`).
    pub fn chunk_at_idx(&self, idx: usize) -> (&'a [M], usize, usize) {
        self.try_chunk_at_idx(idx).unwrap()
    }

    /// Returns the chunk containing the given char index.
    ///
    /// Also returns the byte and char indices of the beginning of the chunk
    /// and the index of the line that the chunk starts on.
    ///
    /// Note: for convenience, a one-past-the-end `char_idx` returns the last
    /// chunk of the `RopeSlice`.
    ///
    /// The return value is organized as
    /// `(chunk, chunk_byte_idx, chunk_char_idx, chunk_line_idx)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_idx` is out of bounds (i.e. `char_idx > len_chars()`).
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
    pub fn slice<R>(&self, char_range: R) -> Self
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
                let start_byte = width_to_idx(text, start);
                let end_byte = width_to_idx(text, end);
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
    pub fn byte_slice<R>(&self, byte_range: R) -> Self
    where
        R: RangeBounds<usize>,
    {
        match self.get_byte_slice_impl(byte_range) {
            Ok(s) => return s,
            Err(e) => panic!("byte_slice(): {}", e),
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
    /// byte `byte_idx`.
    ///
    /// If `byte_idx == len_bytes()` then an iterator at the end of the
    /// `RopeSlice` is created (i.e. `next()` will return `None`).
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_idx` is out of bounds (i.e. `byte_idx > len_bytes()`).
    #[inline]
    pub fn iter_at(&self, byte_idx: usize) -> Iter<'a, M> {
        if let Some(out) = self.get_iter_at(byte_idx) {
            out
        } else {
            panic!(
                "Attempt to index past end of RopeSlice: byte index {}, RopeSlice byte length {}",
                byte_idx,
                self.total_len()
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
    /// iterator starting at the byte containing `byte_idx`.
    ///
    /// Also returns the byte and char indices of the beginning of the first
    /// chunk to be yielded, and the index of the line that chunk starts on.
    ///
    /// If `byte_idx == len_bytes()` an iterator at the end of the `RopeSlice`
    /// (yielding `None` on a call to `next()`) is created.
    ///
    /// The return value is organized as
    /// `(iterator, chunk_byte_idx, chunk_char_idx, chunk_line_idx)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_idx` is out of bounds (i.e. `byte_idx > len_bytes()`).
    #[inline]
    pub fn chunks_at_idx(&self, idx: usize) -> (Chunks<'a, M>, usize, usize) {
        if let Some(out) = self.get_chunks_at_idx(idx) {
            out
        } else {
            panic!(
                "Attempt to index past end of RopeSlice: byte index {}, RopeSlice byte length {}",
                idx,
                self.total_len()
            );
        }
    }

    /// Creates an iterator over the chunks of the `RopeSlice`, with the
    /// iterator starting on the chunk containing `char_idx`.
    ///
    /// Also returns the byte and char indices of the beginning of the first
    /// chunk to be yielded, and the index of the line that chunk starts on.
    ///
    /// If `char_idx == len_chars()` an iterator at the end of the `RopeSlice`
    /// (yielding `None` on a call to `next()`) is created.
    ///
    /// The return value is organized as
    /// `(iterator, chunk_byte_idx, chunk_char_idx, chunk_line_idx)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_idx` is out of bounds (i.e. `char_idx > len_chars()`).
    #[inline]
    pub fn chunks_at_width(&self, width_idx: usize) -> (Chunks<'a, M>, usize, usize) {
        if let Some(out) = self.get_chunks_at_width(width_idx) {
            out
        } else {
            panic!(
                "Attempt to index past end of RopeSlice: char index {}, RopeSlice char length {}",
                width_idx,
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
    pub fn try_idx_to_width(&self, byte_idx: usize) -> Result<usize> {
        // Bounds check
        if byte_idx <= self.total_len() {
            let (chunk, b, c) = self.chunk_at_idx(byte_idx);
            Ok(c + idx_to_width(chunk, byte_idx - b))
        } else {
            Err(Error::ByteIndexOutOfBounds(byte_idx, self.total_len()))
        }
    }

    /// Non-panicking version of [`char_to_byte()`](RopeSlice::char_to_byte).
    #[inline]
    pub fn try_width_to_idx(&self, width: usize) -> Result<usize> {
        // Bounds check
        if width <= self.width() {
            let (chunk, b, c) = self.chunk_at_width(width);
            Ok(b + width_to_idx(chunk, width - c))
        } else {
            Err(Error::CharIndexOutOfBounds(width, self.width()))
        }
    }

    /// Non-panicking version of [`get_byte()`](RopeSlice::get_byte).
    #[inline]
    pub fn get_from_idx(&self, idx: usize) -> Option<&M> {
        // Bounds check
        if idx < self.total_len() {
            let (chunk, chunk_byte_idx, _) = self.chunk_at_idx(idx);
            let chunk_rel_byte_idx = idx - chunk_byte_idx;
            Some(&chunk[chunk_rel_byte_idx])
        } else {
            None
        }
    }

    /// Non-panicking version of [`char()`](RopeSlice::char).
    #[inline]
    pub fn get_from_width(&self, width: usize) -> Option<&M> {
        // Bounds check
        if width < self.width() {
            let (chunk, _, chunk_char_idx) = self.chunk_at_width(width);
            let byte_idx = width_to_idx(chunk, width - chunk_char_idx);
            Some(&chunk[byte_idx])
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunk_at_byte()`](RopeSlice::chunk_at_byte).
    pub fn try_chunk_at_idx(&self, idx: usize) -> Result<(&'a [M], usize, usize)> {
        // Bounds check
        if idx <= self.total_len() {
            match *self {
                RopeSlice(RSEnum::Full {
                    node,
                    start_info,
                    end_info,
                }) => {
                    // Get the chunk.
                    let (chunk, chunk_start_info) =
                        node.get_chunk_at_idx(idx + start_info.len as usize);

                    // Calculate clipped start/end byte indices within the chunk.
                    let chunk_start_byte_idx = start_info.len.saturating_sub(chunk_start_info.len);
                    let chunk_end_byte_idx =
                        (chunk.len() as Count).min(end_info.len - chunk_start_info.len);

                    // Return the clipped chunk and byte offset.
                    Ok((
                        &chunk[chunk_start_byte_idx as usize..chunk_end_byte_idx as usize],
                        chunk_start_info.len.saturating_sub(start_info.len) as usize,
                        chunk_start_info.width.saturating_sub(start_info.width) as usize,
                    ))
                }
                RopeSlice(RSEnum::Light { slice: text, .. }) => Ok((text, 0, 0)),
            }
        } else {
            Err(Error::ByteIndexOutOfBounds(idx, self.total_len()))
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
                        node.get_chunk_at_width(width + start_info.width as usize);

                    // Calculate clipped start/end byte indices within the chunk.
                    let chunk_start_byte_idx = start_info.len.saturating_sub(chunk_start_info.len);
                    let chunk_end_byte_idx =
                        (chunk.len() as Count).min(end_info.len - chunk_start_info.len);

                    // Return the clipped chunk and byte offset.
                    Some((
                        &chunk[chunk_start_byte_idx as usize..chunk_end_byte_idx as usize],
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
    pub fn get_slice<R>(&self, width_range: R) -> Option<RopeSlice<'a, M>>
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
                    let start_byte = width_to_idx(text, start);
                    let end_byte = width_to_idx(text, end);
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

    /// Non-panicking version of [`byte_slice()`](RopeSlice::byte_slice).
    pub fn get_byte_slice<R>(&self, byte_range: R) -> Option<RopeSlice<'a, M>>
    where
        R: RangeBounds<usize>,
    {
        self.get_byte_slice_impl(byte_range).ok()
    }

    pub(crate) fn get_byte_slice_impl<R>(&self, byte_range: R) -> Result<RopeSlice<'a, M>>
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
                    return Err(Error::ByteRangeInvalid(s, e));
                } else if e > self.total_len() {
                    return Err(Error::ByteRangeOutOfBounds(
                        start_range,
                        end_range,
                        self.total_len(),
                    ));
                }
            }
            (Some(s), None) => {
                if s > self.total_len() {
                    return Err(Error::ByteRangeOutOfBounds(
                        start_range,
                        end_range,
                        self.total_len(),
                    ));
                }
            }
            (None, Some(e)) => {
                if e > self.total_len() {
                    return Err(Error::ByteRangeOutOfBounds(
                        start_range,
                        end_range,
                        self.total_len(),
                    ));
                }
            }
        }

        let (start, end) = (
            start_range.unwrap_or(0),
            end_range.unwrap_or_else(|| self.total_len()),
        );

        match *self {
            RopeSlice(RSEnum::Full {
                node, start_info, ..
            }) => RopeSlice::new_with_idx_range(
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
    pub fn get_iter_at(&self, idx: usize) -> Option<Iter<'a, M>> {
        // Bounds check
        if idx <= self.total_len() {
            match *self {
                RopeSlice(RSEnum::Full {
                    node,
                    start_info,
                    end_info,
                }) => Some(Iter::new_with_range_at(
                    node,
                    start_info.len as usize + idx,
                    (start_info.len as usize, end_info.len as usize),
                    (start_info.width as usize, end_info.width as usize),
                )),
                RopeSlice(RSEnum::Light { slice: text, .. }) => {
                    Some(Iter::from_slice_at(text, idx))
                }
            }
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunks_at_byte()`](RopeSlice::chunks_at_byte).
    #[inline]
    pub fn get_chunks_at_idx(
        &self,
        byte_idx: usize,
    ) -> Option<(Chunks<'a, M>, usize, usize)> {
        // Bounds check
        if byte_idx <= self.total_len() {
            match *self {
                RopeSlice(RSEnum::Full {
                    node,
                    start_info,
                    end_info,
                }) => {
                    let (chunks, chunk_byte_idx, chunk_char_idx) =
                        Chunks::new_with_range_at_idx(
                            node,
                            byte_idx + start_info.len as usize,
                            (start_info.len as usize, end_info.len as usize),
                            (start_info.width as usize, end_info.width as usize),
                        );

                    Some((
                        chunks,
                        chunk_byte_idx.saturating_sub(start_info.len as usize),
                        chunk_char_idx.saturating_sub(start_info.width as usize),
                    ))
                }
                RopeSlice(RSEnum::Light {
                    slice: text,
                    count: char_count,
                    ..
                }) => {
                    let chunks = Chunks::from_slice(text, byte_idx == text.len());

                    if byte_idx == text.len() {
                        Some((
                            chunks,
                            text.len(),
                            char_count as usize,
                        ))
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
                    let (chunks, chunk_byte_idx, chunk_char_idx) = Chunks::new_with_range_at_width(
                        node,
                        width + start_info.width as usize,
                        (start_info.len as usize, end_info.len as usize),
                        (start_info.width as usize, end_info.width as usize),
                    );

                    Some((
                        chunks,
                        chunk_byte_idx.saturating_sub(start_info.len as usize),
                        chunk_char_idx.saturating_sub(start_info.width as usize),
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
        let mut vec = Vec::with_capacity(s.total_len());
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
        if self.total_len() != other.total_len() {
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
                if self.total_len() != other.len() {
                    return false;
                }

                let mut idx = 0;
                for chunk in self.chunks() {
                    if chunk != &other[idx..(idx + chunk.len())] {
                        return false;
                    }
                    idx += chunk.len();
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
        *self == other.slice(..)
    }
}

impl<'a, M> std::cmp::PartialEq<RopeSlice<'a, M>> for Rope<M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &RopeSlice<'a, M>) -> bool {
        self.slice(..) == *other
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

        self.total_len().cmp(&other.total_len())
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
        rope::Measurable,
        slice_utils::{idx_to_width, width_to_idx},
        Rope,
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Test {
        Foo,
        Bar,
        Baz(usize),
        Qux,
        Quux,
    }

    impl Measurable for Test {
        fn width(&self) -> usize {
            match self {
                Test::Foo => 1,
                Test::Bar => 2,
                Test::Baz(width) => *width,
                Test::Qux => 0,
                Test::Quux => 0,
            }
        }
    }

    use self::Test::*;
    // 7 elements, total width of 10.
    const NATURAL_WIDTH: &[Test] = &[Foo, Bar, Foo, Bar, Foo, Bar, Foo];
    // 7 elements, total width of 40.
    const VAR_WIDTH: &[Test] = &[Foo, Bar, Baz(4), Bar, Baz(30), Foo];
    // 7 elements, total width of 5.
    const ZERO_WIDTH: &[Test] = &[Qux, Quux, Bar, Qux, Quux, Baz(3), Qux];

    #[test]
    fn len_bytes_01() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        let s = r.slice(7..98);
        assert_eq!(s.total_len(), 105);
    }

    #[test]
    fn len_bytes_02() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        let s = r.slice(43..43);
        assert_eq!(s.total_len(), 0);
    }

    #[test]
    fn len_chars_01() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        let s = r.slice(7..98);
        assert_eq!(s.width(), 91);
    }

    #[test]
    fn len_chars_02() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        let s = r.slice(43..43);
        assert_eq!(s.width(), 0);
    }

    #[test]
    fn byte_to_char_01() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        let s = r.slice(88..102);

        assert_eq!(0, s.idx_to_width(0));
        assert_eq!(1, s.idx_to_width(1));
        assert_eq!(2, s.idx_to_width(2));

        assert_eq!(3, s.idx_to_width(3));
        assert_eq!(3, s.idx_to_width(4));
        assert_eq!(3, s.idx_to_width(5));

        assert_eq!(4, s.idx_to_width(6));
        assert_eq!(4, s.idx_to_width(7));
        assert_eq!(4, s.idx_to_width(8));

        assert_eq!(13, s.idx_to_width(33));
        assert_eq!(13, s.idx_to_width(34));
        assert_eq!(13, s.idx_to_width(35));
        assert_eq!(14, s.idx_to_width(36));
    }

    #[test]
    fn char_to_byte_01() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.slice(88..102);

        assert_eq!(0, s.width_to_idx(0));
        assert_eq!(1, s.width_to_idx(1));
        assert_eq!(2, s.width_to_idx(2));

        assert_eq!(3, s.width_to_idx(3));
        assert_eq!(6, s.width_to_idx(4));
        assert_eq!(33, s.width_to_idx(13));
        assert_eq!(36, s.width_to_idx(14));
    }

    #[test]
    fn byte_01() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.slice(34..100);

        assert_eq!(s.from_idx(0), b't');
        assert_eq!(s.idx(10), b' ');

        // UTF-8 encoding of 'な'.
        assert_eq!(s.from_idx(s.total_len() - 3), 0xE3);
        assert_eq!(s.from_idx(s.total_len() - 2), 0x81);
        assert_eq!(s.from_idx(s.total_len() - 1), 0xAA);
    }

    #[test]
    #[should_panic]
    fn byte_02() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.slice(34..100);
        s.from_idx(s.total_len());
    }

    #[test]
    #[should_panic]
    fn byte_03() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.slice(42..42);
        s.from_idx(0);
    }

    #[test]
    fn char_01() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.slice(34..100);

        // t's \
        // a fine day, isn't it?  Aren't you glad \
        // we're alive?  こんにちは、みんな

        assert_eq!(s.from_width(0), 't');
        assert_eq!(s.from_width(10), ' ');
        assert_eq!(s.from_width(18), 'n');
        assert_eq!(s.from_width(65), 'な');
    }

    #[test]
    #[should_panic]
    fn char_02() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.slice(34..100);
        s.from_width(66);
    }

    #[test]
    #[should_panic]
    fn char_03() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.slice(43..43);
        s.from_width(0);
    }

    #[test]
    fn chunk_at_byte() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.slice(34..96);
        let text = &VAR_WIDTH[34..112];
        // "'s a fine day, isn't it?\nAren't you glad \
        //  we're alive?\nこんにちは、みん"

        let mut t = text;
        let mut prev_chunk = &[];
        for i in 0..s.total_len() {
            let (chunk, b, c) = s.chunk_at_idx(i);
            assert_eq!(c, idx_to_width(text, b));
            if chunk != prev_chunk {
                assert_eq!(chunk, &t[..chunk.len()]);
                t = &t[chunk.len()..];
                prev_chunk = chunk;
            }

            let c1 = {
                let i2 = idx_to_width(text, i);
                text.chars().nth(i2).unwrap()
            };
            let c2 = {
                let i2 = i - b;
                let i3 = idx_to_width(chunk, i2);
                chunk.chars().nth(i3).unwrap()
            };
            assert_eq!(c1, c2);
        }

        assert_eq!(t.len(), 0);
    }

    #[test]
    fn chunk_at_char() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.slice(34..96);
        let text = &VAR_WIDTH[34..112];
        // "'s a fine day, isn't it?\nAren't you glad \
        //  we're alive?\nこんにちは、みん"

        let mut t = text;
        let mut prev_chunk = "";
        for i in 0..s.width() {
            let (chunk, b, c) = s.chunk_at_width(i);
            assert_eq!(b, width_to_idx(text, c));
            if chunk != prev_chunk {
                assert_eq!(chunk, &t[..chunk.len()]);
                t = &t[chunk.len()..];
                prev_chunk = chunk;
            }

            let c1 = text.chars().nth(i).unwrap();
            let c2 = {
                let i2 = i - c;
                chunk.chars().nth(i2).unwrap()
            };
            assert_eq!(c1, c2);
        }
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn slice_01() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s1 = r.slice(..);

        let s2 = s1.slice(..);

        assert_eq!(VAR_WIDTH, s2);
    }

    #[test]
    fn slice_02() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s1 = r.slice(5..43);

        let s2 = s1.slice(3..25);

        assert_eq!(&VAR_WIDTH[8..30], s2);
    }

    #[test]
    fn slice_03() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s1 = r.slice(31..97);

        let s2 = s1.slice(7..64);

        assert_eq!(&VAR_WIDTH[38..103], s2);
    }

    #[test]
    fn slice_04() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s1 = r.slice(5..43);

        let s2 = s1.slice(21..21);

        assert!(s2.is_light());
        assert_eq!("", s2);
    }

    #[test]
    fn slice_05() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s1 = r.slice(5..98);
        for i in 0..(s1.width() - 1) {
            let s2 = s1.slice(i..(i + 1));
            assert!(s2.is_light());
        }
    }

    #[test]
    #[should_panic]
    fn slice_06() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.slice(5..43);

        #[allow(clippy::reversed_empty_ranges)]
        s.slice(21..20); // Wrong ordering on purpose.
    }

    #[test]
    #[should_panic]
    fn slice_07() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.slice(5..43);

        s.slice(37..39);
    }

    #[test]
    fn byte_slice_01() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s1 = r.byte_slice(..);

        let s2 = s1.byte_slice(..);

        assert_eq!(VAR_WIDTH, s2);
    }

    #[test]
    fn byte_slice_02() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s1 = r.byte_slice(50..118);

        let s2 = s1.byte_slice(3..25);

        assert_eq!(&VAR_WIDTH[53..75], s2);
    }

    #[test]
    fn byte_slice_03() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s1 = r.byte_slice(50..118);

        let s2 = s1.byte_slice(7..65);

        assert_eq!(&VAR_WIDTH[57..115], s2);
    }

    #[test]
    fn byte_slice_04() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s1 = r.byte_slice(50..118);

        let s2 = s1.byte_slice(21..21);

        assert!(s2.is_light());
        assert_eq!("", s2);
    }

    #[test]
    fn byte_slice_05() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s1 = r.byte_slice(4..86);
        for i in 0..(s1.len_idx() - 1) {
            let s2 = s1.byte_slice(i..(i + 1));
            assert!(s2.is_light());
        }
    }

    #[test]
    #[should_panic]
    fn byte_slice_06() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.byte_slice(50..118);

        #[allow(clippy::reversed_empty_ranges)]
        s.byte_slice(21..20); // Wrong ordering on purpose.
    }

    #[test]
    #[should_panic]
    fn byte_slice_07() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.byte_slice(50..85);

        s.byte_slice(35..36);
    }

    #[test]
    #[should_panic]
    fn byte_slice_08() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.byte_slice(50..118);

        // Not a char boundary.
        s.byte_slice(..43);
    }

    #[test]
    #[should_panic]
    fn byte_slice_09() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.byte_slice(50..118);

        // Not a char boundary.
        s.byte_slice(43..);
    }

    #[test]
    fn eq_str_01() {
        let r = Rope::from_slice(VAR_WIDTH);
        let slice = r.slice(..);

        assert_eq!(slice, VAR_WIDTH);
        assert_eq!(VAR_WIDTH, slice);
    }

    #[test]
    fn eq_str_02() {
        let r = Rope::from_slice(VAR_WIDTH);
        let slice = r.slice(0..20);

        assert_ne!(slice, VAR_WIDTH);
        assert_ne!(VAR_WIDTH, slice);
    }

    #[test]
    fn eq_str_03() {
        let mut r = Rope::from_slice(VAR_WIDTH);
        r.remove(20..21);
        r.insert(20, "z");
        let slice = r.slice(..);

        assert_ne!(slice, VAR_WIDTH);
        assert_ne!(VAR_WIDTH, slice);
    }

    #[test]
    fn eq_str_04() {
        let r = Rope::from_slice(VAR_WIDTH);
        let slice = r.slice(..);
        let s: Vec<Test> = VAR_WIDTH.into();

        assert_eq!(slice, s);
        assert_eq!(s, slice);
    }

    #[test]
    fn eq_rope_slice_01() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.slice(43..43);

        assert_eq!(s, s);
    }

    #[test]
    fn eq_rope_slice_02() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s1 = r.slice(43..97);
        let s2 = r.slice(43..97);

        assert_eq!(s1, s2);
    }

    #[test]
    fn eq_rope_slice_03() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s1 = r.slice(43..43);
        let s2 = r.slice(43..45);

        assert_ne!(s1, s2);
    }

    #[test]
    fn eq_rope_slice_04() {
        let r = Rope::from_slice(VAR_WIDTH);
        let s1 = r.slice(43..45);
        let s2 = r.slice(43..43);

        assert_ne!(s1, s2);
    }

    #[test]
    fn eq_rope_slice_05() {
        let r = Rope::from_slice(&[]);
        let s = r.slice(0..0);

        assert_eq!(s, s);
    }

    #[test]
    fn cmp_rope_slice_01() {
        let r1 = Rope::from_slice(ZERO_WIDTH);
        let r2 = Rope::from_slice(ZERO_WIDTH);
        let s1 = r1.slice(..);
        let s2 = r2.slice(..);

        assert_eq!(s1.cmp(&s2), std::cmp::Ordering::Equal);
        assert_eq!(s1.slice(..24).cmp(&s2), std::cmp::Ordering::Less);
        assert_eq!(s1.cmp(&s2.slice(..24)), std::cmp::Ordering::Greater);
    }

    #[test]
    fn cmp_rope_slice_02() {
        let r1 = Rope::from_slice("abcdefghijklmnzpqrstuvwxyz");
        let r2 = Rope::from_slice("abcdefghijklmnopqrstuvwxyz");
        let s1 = r1.slice(..);
        let s2 = r2.slice(..);

        assert_eq!(s1.cmp(&s2), std::cmp::Ordering::Greater);
        assert_eq!(s2.cmp(&s1), std::cmp::Ordering::Less);
    }

    #[test]
    fn to_string_01() {
        let r = Rope::from_slice(VAR_WIDTH);
        let slc = r.slice(..);
        let s: String = slc.into();

        assert_eq!(r, s);
        assert_eq!(slc, s);
    }

    #[test]
    fn to_string_02() {
        let r = Rope::from_slice(VAR_WIDTH);
        let slc = r.slice(0..24);
        let s: String = slc.into();

        assert_eq!(slc, s);
    }

    #[test]
    fn to_string_03() {
        let r = Rope::from_slice(VAR_WIDTH);
        let slc = r.slice(13..89);
        let s: String = slc.into();

        assert_eq!(slc, s);
    }

    #[test]
    fn to_string_04() {
        let r = Rope::from_slice(VAR_WIDTH);
        let slc = r.slice(13..41);
        let s: String = slc.into();

        assert_eq!(slc, s);
    }

    #[test]
    fn to_cow_01() {
        use std::borrow::Cow;
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.slice(13..83);
        let cow: Cow<str> = s.into();

        assert_eq!(s, cow);
    }

    #[test]
    fn to_cow_02() {
        use std::borrow::Cow;
        let r = Rope::from_slice(VAR_WIDTH);
        let s = r.slice(13..14);
        let cow: Cow<str> = r.slice(13..14).into();

        // Make sure it's borrowed.
        if let Cow::Owned(_) = cow {
            panic!("Small Cow conversions should result in a borrow.");
        }

        assert_eq!(s, cow);
    }
}
