#![allow(
    incomplete_features,
    clippy::arc_with_non_send_sync,
    clippy::too_many_arguments
)]
#![feature(generic_const_exprs)]
//! AnyRope is an arbitrary data rope for Rust.
//!
//! AnyRope's [`Rope<M>`] contains elements `M` that implement [`Measurable`], a
//! trait that assigns an arbitrary "measure" to each element, through the
//! [`measure()`][Measurable::measure] function. AnyRope can then use these
//! "measures" to retrieve and iterate over elements in any given "measure" from
//! the beginning of the [`Rope<M>`].
//!
//! Keep in mind that the "measure" does not correspond to the actual size of a
//! type in bits or bytes, but is instead decided by the implementor, and can be
//! whatever value they want.
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
//! in which the tags either tell you to print normally, print in red,
//! underline, or skip:
//!
//! ```rust
//! #![feature(generic_const_exprs)]
//! # use std::io::Result;
//! use std::fs::File;
//! use std::io::{BufReader, BufWriter};
//! use any_rope::{Rope, Measurable};
//!
//! // A simple tag structure that our program can understand.
//! #[derive(Clone, Copy, PartialEq, Eq)]
//! enum Tag {
//!     InRed,
//!     UnderLine,
//!     Normal,
//!     // The `usize` in here represents an amount of characters that won't change
//!     // the color of the text.
//!     Skip(usize)
//! }
//!
//! impl Measurable for Tag {
//!     type Measure = usize;
//!
//!     fn measure(&self) -> Self::Measure {
//!         match self {
//!             // The coloring tags are only meant to color, not to "move forward".
//!             Tag::InRed | Tag::UnderLine | Tag::Normal => 0,
//!             // The Skip tag represents an amount of characters in which no
//!             // tags are applied.
//!             Tag::Skip(amount) => *amount
//!         }
//!     }
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
//! let mut tags_iter = my_tagger.iter().peekable();
//! for (cur_index, ch) in my_str.chars().enumerate() {
//!     // The while let loop here is a useful way to activate all tags within the same
//!     // character. Note the sequence of [.., InRed, UnderLine, ..], both of which have
//!     // a measure of 0. This means that both would be triggered before moving on to the next
//!     // character.
//!     while let Some((index, tag)) = tags_iter.peek() {
//!         // The returned index is always the measure where an element began. In this
//!         // case, `tags_iter.peek()` would return `Some((0, Skip(5)))`, and then
//!         // `Some((5, InRed))`.
//!         if *index == cur_index {
//!             activate_tag(tag);
//!             tags_iter.next();
//!         } else {
//!             break;
//!         }
//!     }
//!
//!     print!("{}", ch);
//! }
//! ```
//!
//! An example can be found in the `examples` directory, detailing a "search and
//! replace" functionality for [`Rope<M>`].
//!
//! # Low-level APIs
//!
//! AnyRope also provides access to some of its low-level APIs, enabling client
//! code to efficiently work with a [`Rope<M>`]'s data and implement new
//! functionality.  The most important of those API's are:
//!
//! - The [`chunk_at_*()`][Rope::chunk_at_measure] chunk-fetching methods of
//!   [`Rope<M>`] and [`RopeSlice<M>`].
//! - The [`Chunks`](iter::Chunks) iterator.
//! - The functions in `slice_utils` for operating on [`&[M]`][Measurable]
//!   slices.
//!
//! As a reminder, if you notice similarities with the AnyRope crate, it is
//! because this is a heavily modified fork of it.
#![allow(
    clippy::collapsible_if,
    clippy::inline_always,
    clippy::needless_return,
    clippy::redundant_field_names,
    clippy::type_complexity
)]

mod rope;
mod rope_builder;
mod slice;
mod slice_utils;
mod tree;

pub mod iter;

use std::{
    cmp::Ordering,
    fmt::Debug,
    ops::{Add, Bound, Range, RangeFrom, RangeFull, RangeTo, Sub},
};

pub use crate::{
    rope::Rope,
    rope_builder::RopeBuilder,
    slice::RopeSlice,
    tree::{max_children, max_len},
};

/// A trait defining a comparison that must panic if there is ever ambiguity
/// about the ordering in a given struct.
pub trait FallibleOrd {
    /// A comparison that can panic.
    ///
    /// This comparison should panic with the [`unreachable!`] macro, since the
    /// panicking code is supposed to be unreachable from within `any-rope`.
    ///
    /// Here's an example of how this function should be implemented:
    ///
    /// ```rust
    /// # use std::cmp::Ordering;
    /// # use any_rope::FallibleOrd;
    /// struct TwoUsizes(usize, usize);
    ///
    /// impl FallibleOrd for TwoUsizes {
    ///     fn fallible_cmp(&self, other: &Self) -> Ordering {
    ///         use Ordering::*;
    ///         let cmp_0 = self.0.cmp(&other.0);
    ///         let cmp_1 = self.1.cmp(&other.1);
    ///
    ///         match (cmp_0, cmp_1) {
    ///             (Less, Less) => Less,
    ///             (Less, Equal) => Less,
    ///             (Equal, Less) => Less,
    ///             (Equal, Equal) => Equal,
    ///             (Equal, Greater) => Greater,
    ///             (Greater, Equal) => Greater,
    ///             (Greater, Greater) => todo!(),
    ///             (Less, Greater) | (Greater, Less) => unreachable!(
    ///                 "Ambiguous comparison, report this as a bug on the \
    ///                  AnyRope crate"
    ///             ),
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// Note that, if all values are equal, then the comparison is equal. If at
    /// least one value is greater or lesse, then the comparison follows that
    /// value. But if there are greater and lesser values at the same time, then
    /// something has failed within `any-rope` itself, and that should be
    /// reported as a bug.
    fn fallible_cmp(&self, other: &Self) -> Ordering;
}

/// A object that has a user defined size, that can be interpreted by a
/// [`Rope<M>`].
pub trait Measurable: Clone + Copy + PartialEq + Eq {
    /// This type is what will be used to query, iterate, modify, and slice up
    /// the [`Rope`].
    ///
    /// It needs to be light, since it will be heavily utilized from within
    /// `any-rope`, and needs to be malleable, such that it can be compared,
    /// added and subtracted at will by the rope.
    ///
    /// Normally, in order to query, iterate, or modify the rope, the various
    /// methods take in a comparator function. This function offers flexibility
    /// in how one searches within the rope, for example:
    ///
    /// ```rust
    /// # use std::{
    /// #     cmp::Ordering,
    /// #     ops::{Add, Sub},
    /// # };
    /// # use any_rope::{FallibleOrd, Measurable, Rope};
    /// #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    /// struct TwoWidths(usize, usize);
    ///
    /// impl TwoWidths {
    ///     fn first_cmp(&self, other: &Self) -> Ordering {
    ///         self.0.cmp(&other.0)
    ///     }
    ///
    ///     fn second_cmp(&self, other: &Self) -> Ordering {
    ///         self.1.cmp(&other.1)
    ///     }
    /// }
    ///
    /// impl FallibleOrd for TwoWidths {
    ///     fn fallible_cmp(&self, other: &Self) -> Ordering {
    ///         use Ordering::*;
    ///         let cmp_0 = self.0.cmp(&other.0);
    ///         let cmp_1 = self.1.cmp(&other.1);
    ///
    ///         match (cmp_0, cmp_1) {
    ///             (Less, Less) => Less,
    ///             (Less, Equal) => Less,
    ///             (Equal, Less) => Less,
    ///             (Equal, Equal) => Equal,
    ///             (Equal, Greater) => Greater,
    ///             (Greater, Equal) => Greater,
    ///             (Greater, Greater) => todo!(),
    ///             (Less, Greater) | (Greater, Less) => unreachable!(
    ///                 "Ambiguous comparison, report this as a bug on the \
    ///                  AnyRope crate"
    ///             ),
    ///         }
    ///     }
    /// }
    ///
    /// // ... Implementation of `Add` and `Sub` for `TwoWidths`.
    ///
    /// # impl Add for TwoWidths {
    /// #     type Output = Self;
    ///
    /// #     fn add(self, other: Self) -> Self::Output {
    /// #         Self(self.0 + other.0, self.1 + other.1)
    /// #     }
    /// # }
    /// # impl Sub for TwoWidths {
    /// #     type Output = Self;
    ///
    /// #     fn sub(self, other: Self) -> Self::Output {
    /// #         Self(self.0 - other.0, self.1 - other.1)
    /// #     }
    /// # }
    /// #[derive(Clone, Copy, PartialEq, Eq)]
    /// struct MyStruct(TwoWidths);
    ///
    /// impl Measurable for MyStruct {
    ///     type Measure = TwoWidths;
    ///
    ///     fn measure(&self) -> Self::Measure {
    ///         self.0
    ///     }
    /// }
    /// ```
    ///
    /// With the `TwoWidths::first_cmp` and `TwoWidths::second_cmp`, you can
    /// search with two different variables within the rope, similarly to how
    /// regular ropes let you search by byte, char, or line.
    ///
    /// Note, however, that this type also requires `PartialOrd` and `Ord`. That
    /// is because, internally, `any-rope` needs those comparison functions to
    /// sort itself out. These implementations also cannot be derived by Rust,
    /// and need to be
    type Measure: Debug
        + Default
        + Clone
        + Copy
        + PartialEq
        + Eq
        + Add<Output = Self::Measure>
        + Sub<Output = Self::Measure>
        + FallibleOrd;

    /// The measure of this element, it need not be the actual lenght in bytes,
    /// but just a representative value, to be fed to the [`Rope<M>`].
    fn measure(&self) -> Self::Measure;
}

/// A struct meant for testing and exemplification
///
/// Its [`measure`][Measurable::measure] is always equal to the number within.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Width(pub usize);

impl PartialOrd for Width {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Width {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.fallible_cmp(&other.0)
    }
}

impl FallibleOrd for usize {
    fn fallible_cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

impl Measurable for Width {
    type Measure = usize;

    fn measure(&self) -> Self::Measure {
        self.0
    }
}

//==============================================================
// Error reporting types.

/// AnyRope's result type.
pub type Result<T, M> = std::result::Result<T, Error<M>>;

/// AnyRope's error type.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum Error<M: Measurable> {
    /// Indicates that the passed index was out of bounds.
    ///
    /// Contains the index attempted and the actual length of the
    /// [`Rope<M>`]/[`RopeSlice<M>`], in that order.
    IndexOutOfBounds(usize, usize),

    /// Indicates that the passed measure was out of bounds.
    ///
    /// Contains the index attempted and the actual measure of the
    /// [`Rope<M>`]/[`RopeSlice<M>`], in that order.
    MeasureOutOfBounds(M::Measure, M::Measure),

    /// Indicates that a reversed index range (end < start) was encountered.
    ///
    /// Contains the [start, end) indices of the range, in that order.
    IndexRangeInvalid(usize, usize),

    /// Indicates that a reversed measure range (end < start) was
    /// encountered.
    ///
    /// Contains the [start, end) measures of the range, in that order.
    MeasureRangeInvalid(Option<M::Measure>, Option<M::Measure>),

    /// Indicates that the passed index range was partially or fully out of
    /// bounds.
    ///
    /// Contains the [start, end) indices of the range and the actual
    /// length of the [`Rope<M>`]/[`RopeSlice<M>`], in that order.
    /// When either the start or end are [`None`], that indicates a half-open
    /// range.
    IndexRangeOutOfBounds(Option<usize>, Option<usize>, usize),

    /// Indicates that the passed measure range was partially or fully out of
    /// bounds.
    ///
    /// Contains the [start, end) measures of the range and the actual
    /// measure of the [`Rope<M>`]/[`RopeSlice<M>`], in that order.
    /// When either the start or end are [`None`], that indicates a half-open
    /// range.
    MeasureRangeOutOfBounds(Option<M::Measure>, Option<M::Measure>, M::Measure),
}

impl<M: Measurable> std::fmt::Debug for Error<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Error::IndexOutOfBounds(index, len) => {
                write!(
                    f,
                    "Index out of bounds: index {}, Rope/RopeSlice length {}",
                    index, len
                )
            }
            Error::MeasureOutOfBounds(measure, max) => {
                write!(
                    f,
                    "Measure out of bounds: measure {:?}, Rope/RopeSlice measure {:?}",
                    measure, max
                )
            }
            Error::IndexRangeInvalid(start_index, end_index) => {
                write!(
                    f,
                    "Invalid index range {}..{}: start must be <= end",
                    start_index, end_index
                )
            }
            Error::MeasureRangeInvalid(start_measure, end_measure) => {
                write!(
                    f,
                    "Invalid measure range {:?}..{:?}: start must be <= end",
                    start_measure, end_measure
                )
            }
            Error::IndexRangeOutOfBounds(start, end, len) => {
                write!(f, "Index range out of bounds: index range ")?;
                write_range(f, start, end)?;
                write!(f, ", Rope/RopeSlice byte length {}", len)
            }
            Error::MeasureRangeOutOfBounds(start, end, measure) => {
                write!(f, "Measure range out of bounds: measure range ")?;
                write_range(f, start, end)?;
                write!(f, ", Rope/RopeSlice measure {:?}", measure)
            }
        }
    }
}

impl<M: Measurable> std::error::Error for Error<M> {}

impl<M: Measurable> std::fmt::Display for Error<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Just re-use the debug impl.
        std::fmt::Debug::fmt(self, f)
    }
}

fn write_range<T: Debug>(
    f: &mut std::fmt::Formatter<'_>,
    start_idx: Option<T>,
    end_idx: Option<T>,
) -> std::fmt::Result {
    match (start_idx, end_idx) {
        (None, None) => write!(f, ".."),
        (None, Some(end)) => write!(f, "..{:?}", end),
        (Some(start), None) => write!(f, "{:?}..", start),
        (Some(start), Some(end)) => write!(f, "{:?}..{:?}", start, end),
    }
}

//==============================================================
// Range handling utilities.

#[inline(always)]
fn start_bound_to_num(b: Bound<&usize>) -> Option<usize> {
    match b {
        Bound::Included(n) => Some(*n),
        Bound::Excluded(n) => Some(*n + 1),
        Bound::Unbounded => None,
    }
}

#[inline(always)]
fn end_bound_to_num(b: Bound<&usize>) -> Option<usize> {
    match b {
        Bound::Included(n) => Some(*n + 1),
        Bound::Excluded(n) => Some(*n),
        Bound::Unbounded => None,
    }
}

mod hidden {
    use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

    pub trait Internal {}

    impl<T> Internal for Range<T> {}
    impl<T> Internal for RangeFrom<T> {}
    impl<T> Internal for RangeTo<T> {}
    impl Internal for RangeFull {}
}

pub trait MeasureRange<M: Measurable>: hidden::Internal {
    fn start_bound(&self) -> Option<&M::Measure>;

    fn end_bound(&self) -> Option<&M::Measure>;
}

impl<M: Measurable> MeasureRange<M> for Range<M::Measure> {
    fn start_bound(&self) -> Option<&M::Measure> {
        Some(&self.start)
    }

    fn end_bound(&self) -> Option<&M::Measure> {
        Some(&self.end)
    }
}

impl<M: Measurable> MeasureRange<M> for RangeFrom<M::Measure> {
    fn start_bound(&self) -> Option<&M::Measure> {
        Some(&self.start)
    }

    fn end_bound(&self) -> Option<&M::Measure> {
        None
    }
}

impl<M: Measurable> MeasureRange<M> for RangeTo<M::Measure> {
    fn start_bound(&self) -> Option<&M::Measure> {
        None
    }

    fn end_bound(&self) -> Option<&M::Measure> {
        Some(&self.end)
    }
}

impl<M: Measurable> MeasureRange<M> for RangeFull {
    fn start_bound(&self) -> Option<&M::Measure> {
        None
    }

    fn end_bound(&self) -> Option<&M::Measure> {
        None
    }
}

#[inline(always)]
fn measures_from_range<M: Measurable>(
    range: &impl MeasureRange<M>,
    limit: M::Measure,
) -> Result<(M::Measure, M::Measure), M> {
    #[cfg(debug_assertions)]
    {
        if let Some(start) = range.start_bound() {
            assert_default_is_lowest::<M>(start);
        }
        if let Some(end) = range.end_bound() {
            assert_default_is_lowest::<M>(end);
        }
    }

    match (range.start_bound(), range.end_bound()) {
        (None, None) => Ok((M::Measure::default(), limit)),
        (None, Some(end)) => {
            if end.fallible_cmp(&limit).is_gt() {
                Err(Error::MeasureRangeOutOfBounds(None, Some(*end), limit))
            } else {
                Ok((M::Measure::default(), *end))
            }
        }
        (Some(start), None) => {
            if start.fallible_cmp(&limit).is_gt() {
                Err(Error::MeasureRangeOutOfBounds(Some(*start), None, limit))
            } else {
                Ok((*start, limit))
            }
        }

        (Some(start), Some(end)) => {
            if start.fallible_cmp(end).is_gt() {
                Err(Error::MeasureRangeInvalid(Some(*start), Some(*end)))
            } else if end.fallible_cmp(&limit).is_gt() || start.fallible_cmp(&limit).is_gt() {
                Err(Error::MeasureRangeOutOfBounds(
                    Some(*start),
                    Some(*end),
                    limit,
                ))
            } else {
                Ok((*start, *end))
            }
        }
    }
}

//==============================================================
// Other utilities.

fn assert_default_is_lowest<M: Measurable>(cmp: &M::Measure) {
    assert!(
        M::Measure::default().fallible_cmp(cmp).is_le(),
        "{:?} (Measure::default()) is supposed to be the smallest possible value for \
         Measurable::Measure, and yet {:?} is smaller",
        M::Measure::default(),
        cmp
    )
}

fn fallible_saturating_sub<T>(lhs: T, rhs: T) -> T
where
    T: Default + FallibleOrd + Sub<Output = T>,
{
    if lhs.fallible_cmp(&rhs).is_le() {
        T::default()
    } else {
        lhs - rhs
    }
}

fn fallible_min<T: FallibleOrd>(lhs: T, rhs: T) -> T {
    if lhs.fallible_cmp(&rhs).is_le() {
        lhs
    } else {
        rhs
    }
}

fn fallible_max<T: FallibleOrd>(lhs: T, rhs: T) -> T {
    if lhs.fallible_cmp(&rhs).is_ge() {
        lhs
    } else {
        rhs
    }
}
