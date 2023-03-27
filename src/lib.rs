#![feature(generic_const_exprs)]
//! AnyRope is an arbitrary data rope for Rust.
//!
//! AnyRope's [`Rope<M>`] contains elements `M` that implement [`Measurable`], a
//! trait that assigns an arbitrary "width" to each element, through the
//! [`width()`][Measurable::width] function. AnyRope can then use these "widths"
//! to retrieve and iterate over elements in any given "width" from the beginning
//! of the [`Rope<M>`].
//!
//! Keep in mind that the "width" does not correspond to the actual size of a type
//! in bits or bytes, but is instead decided by the implementor, and can be whatever
//! value they want.
//!
//! The library is made up of four main components:
//!
//! - [`Rope<M>`]: the main rope type.
//! - [`RopeSlice<M>`]: an immutable view into part of a [`Rope<M>`].
//! - [`iter`]: iterators over [`Rope<M>`]/[`RopeSlice<M>`] data.
//! - [`RopeBuilder<M>`]: an efficient incremental [`Rope<M>`] builder.
//!
//! # A Basic Example
//!
//! Let's say we want create a tagging system that will be applied to text,
//! in which the tags either tell you to print normally, print in red, underline, or skip:
//!
//! ```rust
//! # use std::io::Result;
//! use std::fs::File;
//! use std::io::{BufReader, BufWriter};
//! use any_rope::{Rope, Measurable};
//!
//! // A simple tag structure that our program can understand.
//! #[derive(Clone, Copy)]
//! enum Tag {
//! 	InRed,
//! 	UnderLine,
//! 	Normal,
//! 	// The `usize` in here represents an amount of characters that won't change
//! 	// the color of the text.
//! 	Skip(usize)
//! }
//!
//! impl Measurable for Tag {
//! 	fn width(&self) -> usize {
//! 		match self {
//! 			// The coloring tags are only meant to color, not to "move forward".
//! 			Tag::InRed | Tag::UnderLine | Tag::Normal => 0,
//! 			// The Skip tag represents an amount of characters in which no
//! 			// tags are applied.
//! 			Tag::Skip(amount) => *amount
//! 		}
//! 	}
//! }
//! use Tag::*;
//!
//! # fn activate_tag(tag: &Tag) {}
//! // An `&str` that will be colored.
//! let my_str = "This word will be red!";
//!
//! // Here's what this means:
//! // - Skip 5 characters;
//! // - Change the color to red;
//! // - Start underlining;
//! // - Skip 4 characters;
//! // - Change the rendering back to normal.
//! let my_tagger = Rope::from_slice(&[Skip(5), InRed, UnderLine, Skip(4), Normal]);
//! // Do note that Tag::Skip only represents characters because we are also iterating
//! // over a `Chars` iterator, and have chosen to do so.
//!
//!	let mut tags_iter = my_tagger.iter().peekable();
//! for (cur_index, ch) in my_str.chars().enumerate() {
//! 	// The while let loop here is a useful way to activate all tags within the same
//! 	// character. Note the sequence of [.., InRed, UnderLine, ..], both of which have
//! 	// a width of 0. This means that both would be triggered before moving on to the next
//! 	// character.
//!		while let Some((index, tag)) = tags_iter.peek() {
//!			// The returned index is always the width where an element began. In this
//!			// case, `tags_iter.peek()` would return `Some((0, Skip(5)))`, and then
//!			// `Some((5, InRed))`.
//!			if *index == cur_index {
//!				activate_tag(tag);
//!				tags_iter.next();
//!			} else {
//!				break;
//!			}
//!		}
//!
//!		print!("{}", ch);
//! }
//! ```
//!
//! An example can be found in the `examples` directory, detailing a "search and replace"
//! functionality for [`Rope<M>`].
//!
//! # Low-level APIs
//!
//! AnyRope also provides access to some of its low-level APIs, enabling client
//! code to efficiently work with a [`Rope<M>`]'s data and implement new
//! functionality.  The most important of those API's are:
//!
//! - The [`chunk_at_*()`][Rope::chunk_at_width]
//!   chunk-fetching methods of [`Rope<M>`] and [`RopeSlice<M>`].
//! - The [`Chunks`](iter::Chunks) iterator.
//! - The functions in `slice_utils` for operating on [`&[M]`][Measurable] slices.
//!
//! As a reminder, if you notice similarities with the AnyRope crate, it is because this
//! is a heavily modified fork of it.
//!
//! # Note about documentation
//!
//! In the documentation of AnyRope, there will be a struct called [`Lipsum`],
//! used to exemplify the features of the crate.
#![allow(clippy::collapsible_if)]
#![allow(clippy::inline_always)]
#![allow(clippy::needless_return)]
#![allow(clippy::redundant_field_names)]
#![allow(clippy::type_complexity)]

mod rope;
mod rope_builder;
mod slice;
mod slice_utils;
mod tree;

pub mod iter;

use std::ops::Bound;

pub use crate::rope::{Measurable, Rope};
pub use crate::rope_builder::RopeBuilder;
pub use crate::slice::RopeSlice;
pub use crate::tree::{max_children, max_len};

/// Simple test struct, useful in making sure that the systems work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Lipsum {
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

//==============================================================
// Error reporting types.

/// AnyRope's result type.
pub type Result<T> = std::result::Result<T, Error>;

/// AnyRope's error type.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum Error {
    /// Indicates that the passed index was out of bounds.
    ///
    /// Contains the index attempted and the actual length of the
    /// [`Rope<M>`]/[`RopeSlice<M>`], in that order.
    IndexOutOfBounds(usize, usize),

    /// Indicates that the passed width was out of bounds.
    ///
    /// Contains the index attempted and the actual width of the
    /// [`Rope<M>`]/[`RopeSlice<M>`], in that order.
    WidthOutOfBounds(usize, usize),

    /// Indicates that a reversed index range (end < start) was encountered.
    ///
    /// Contains the [start, end) indices of the range, in that order.
    IndexRangeInvalid(
        usize, // Start.
        usize, // End.
    ),

    /// Indicates that a reversed width range (end < start) was
    /// encountered.
    ///
    /// Contains the [start, end) widths of the range, in that order.
    WidthRangeInvalid(
        usize, // Start.
        usize, // End.
    ),

    /// Indicates that the passed index range was partially or fully out of bounds.
    ///
    /// Contains the [start, end) indices of the range and the actual
    /// length of the [`Rope<M>`]/[`RopeSlice<M>`], in that order.
    /// When either the start or end are [`None`], that indicates a half-open range.
    IndexRangeOutOfBounds(
        Option<usize>, // Start.
        Option<usize>, // End.
        usize,         // Rope byte length.
    ),

    /// Indicates that the passed width range was partially or fully out of bounds.
    ///
    /// Contains the [start, end) widths of the range and the actual
    /// width of the [`Rope<M>`]/[`RopeSlice<M>`], in that order.
    /// When either the start or end are [`None`], that indicates a half-open range.
    WidthRangeOutOfBounds(
        Option<usize>, // Start.
        Option<usize>, // End.
        usize,         // Rope char length.
    ),
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Error::IndexOutOfBounds(index, len) => {
                write!(
                    f,
                    "Index out of bounds: index {}, Rope/RopeSlice length {}",
                    index, len
                )
            }
            Error::WidthOutOfBounds(index, len) => {
                write!(
                    f,
                    "Width out of bounds: width {}, Rope/RopeSlice char length {}",
                    index, len
                )
            }
            Error::IndexRangeInvalid(start_idx, end_idx) => {
                write!(
                    f,
                    "Invalid index range {}..{}: start must be <= end",
                    start_idx, end_idx
                )
            }
            Error::WidthRangeInvalid(start_idx, end_idx) => {
                write!(
                    f,
                    "Invalid width range {}..{}: start must be <= end",
                    start_idx, end_idx
                )
            }
            Error::IndexRangeOutOfBounds(start_idx_opt, end_idx_opt, len) => {
                write!(f, "Index range out of bounds: index range ")?;
                write_range(f, start_idx_opt, end_idx_opt)?;
                write!(f, ", Rope/RopeSlice byte length {}", len)
            }
            Error::WidthRangeOutOfBounds(start_idx_opt, end_idx_opt, len) => {
                write!(f, "Width range out of bounds: width range ")?;
                write_range(f, start_idx_opt, end_idx_opt)?;
                write!(f, ", Rope/RopeSlice char length {}", len)
            }
        }
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Just re-use the debug impl.
        std::fmt::Debug::fmt(self, f)
    }
}

fn write_range(
    f: &mut std::fmt::Formatter<'_>,
    start_idx: Option<usize>,
    end_idx: Option<usize>,
) -> std::fmt::Result {
    match (start_idx, end_idx) {
        (None, None) => write!(f, ".."),
        (Some(start), None) => write!(f, "{}..", start),
        (None, Some(end)) => write!(f, "..{}", end),
        (Some(start), Some(end)) => write!(f, "{}..{}", start, end),
    }
}

//==============================================================
// Range handling utilities.

#[inline(always)]
pub(crate) fn start_bound_to_num(b: Bound<&usize>) -> Option<usize> {
    match b {
        Bound::Included(n) => Some(*n),
        Bound::Excluded(n) => Some(*n + 1),
        Bound::Unbounded => None,
    }
}

#[inline(always)]
pub(crate) fn end_bound_to_num(b: Bound<&usize>) -> Option<usize> {
    match b {
        Bound::Included(n) => Some(*n + 1),
        Bound::Excluded(n) => Some(*n),
        Bound::Unbounded => None,
    }
}
