use std::borrow::Borrow;
use std::fmt::{Debug, Display};
use std::iter::FromIterator;
use std::ops::Deref;

use crate::rope::Measurable;

use self::inner::LeafSmallVec;

/// A custom small string.  The unsafe guts of this are in `NodeSmallString`
/// further down in this file.
#[derive(Clone, Default)]
#[repr(C)]
pub(crate) struct Leaf<M>(inner::LeafSmallVec<M>)
where
    M: Measurable;

impl<M> Leaf<M>
where
    M: Measurable,
{
    /// Creates a new empty `NodeText`
    #[inline(always)]
    pub fn new() -> Self {
        Leaf(inner::LeafSmallVec::new())
    }

    /// Inserts `string` at `byte_idx` and splits the resulting string in half,
    /// returning the right half.
    ///
    /// Only splits on code point boundaries and will never split CRLF pairs,
    /// so if the whole string is a single code point or CRLF pair, the split
    /// will fail and the returned string will be empty.
    pub fn insert_slice_split(&mut self, split_idx: usize, slice: &[M]) -> Self {
        let tot_len = self.len() + slice.len();
        let mid_idx = tot_len / 2;

        let mut right = Leaf::new();
        if split_idx <= mid_idx {
            right.push_slice(&slice[(mid_idx - split_idx)..]);
            right.push_slice(&self[split_idx..]);
            self.truncate(split_idx);
            self.push_slice(&slice[..(mid_idx - split_idx)]);
        } else {
            right.push_slice(&self[mid_idx..]);
            right.push_slice(slice);
            self.truncate(mid_idx);
        }

        self.0.inline_if_possible();
        right
    }

    /// Appends a `&str` to end the of the `NodeText`.
    pub fn push_slice(&mut self, slice: &[M]) {
        let len = self.len();
        self.0.insert_slice(len, slice);
    }

    /// Appends a `&str` and splits the resulting string in half, returning
    /// the right half.
    ///
    /// Only splits on code point boundaries and will never split CRLF pairs,
    /// so if the whole string is a single code point or CRLF pair, the split
    /// will fail and the returned string will be empty.
    pub fn push_slice_split(&mut self, slice: &[M]) -> Self {
        let len = self.len();
        self.insert_slice_split(len, slice)
    }

    /// Drops the text after byte index `byte_idx`.
    pub fn truncate(&mut self, byte_idx: usize) {
        self.0.truncate(byte_idx);
        self.0.inline_if_possible();
    }

    /// Drops the text before byte index `byte_idx`, shifting the
    /// rest of the text to fill in the space.
    pub fn truncate_front(&mut self, byte_idx: usize) {
        self.0.remove_range(0, byte_idx);
        self.0.inline_if_possible();
    }

    /// Removes the text in the byte index interval `[byte_start, byte_end)`.
    pub fn remove_range(&mut self, byte_start: usize, byte_end: usize) {
        self.0.remove_range(byte_start, byte_end);
        self.0.inline_if_possible();
    }

    /// Splits the `NodeText` at `byte_idx`.
    ///
    /// The left part remains in the original, and the right part is
    /// returned in a new `NodeText`.
    pub fn split_off(&mut self, byte_idx: usize) -> Self {
        let other = Leaf(self.0.split_off(byte_idx));
        self.0.inline_if_possible();
        other
    }
}

impl<M> std::cmp::PartialEq for Leaf<M>
where
    M: Measurable + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        let (s1, s2): (&[M], &[M]) = (self, other);
        s1 == s2
    }
}

impl<'a, M> PartialEq<Leaf<M>> for &'a [M]
where
    M: Measurable + PartialEq,
{
    fn eq(&self, other: &Leaf<M>) -> bool {
        *self == (other as &[M])
    }
}

impl<'a, M> PartialEq<&'a [M]> for Leaf<M>
where
    M: Measurable + PartialEq,
{
    fn eq(&self, other: &&'a [M]) -> bool {
        (self as &[M]) == *other
    }
}

impl<M> std::fmt::Display for Leaf<M>
where
    M: Measurable + Display + Debug,
{
    fn fmt(&self, fm: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        Leaf::deref(self).fmt(fm)
    }
}

impl<M> std::fmt::Debug for Leaf<M>
where
    M: Measurable + Debug,
{
    fn fmt(&self, fm: &mut std::fmt::Formatter) -> std::fmt::Result {
        Leaf::deref(self).fmt(fm)
    }
}

impl<'a, M> From<&'a [M]> for Leaf<M>
where
    M: Measurable,
{
    fn from(value: &'a [M]) -> Self {
        Self(LeafSmallVec::from_slice(value))
    }
}

impl<M> Deref for Leaf<M>
where
    M: Measurable,
{
    type Target = [M];

    fn deref(&self) -> &[M] {
        self.0.as_slice()
    }
}

impl<M> AsRef<[M]> for Leaf<M>
where
    M: Measurable,
{
    fn as_ref(&self) -> &[M] {
        self.0.as_slice()
    }
}

impl<M> Borrow<[M]> for Leaf<M>
where
    M: Measurable,
{
    fn borrow(&self) -> &[M] {
        self.0.as_slice()
    }
}

//=======================================================================

/// The unsafe guts of NodeText, exposed through a safe API.
///
/// Try to keep this as small as possible, and implement functionality on
/// NodeText via the safe APIs whenever possible.
mod inner {
    use crate::tree::MAX_BYTES;
    use smallvec::{Array, SmallVec};
    use std::str;

    use super::Measurable;

    /// The backing internal buffer type for `NodeText`.
    #[derive(Copy, Clone)]
    struct BackingArray<M>([M; MAX_BYTES])
    where
        M: Measurable;

    /// We need a very specific size of array, which is not necessarily
    /// supported directly by the impls in the smallvec crate.  We therefore
    /// have to implement this unsafe trait for our specific array size.
    /// TODO: once integer const generics land, and smallvec updates its APIs
    /// to use them, switch over and get rid of this unsafe impl.
    unsafe impl<M> Array for BackingArray<M>
    where
        M: Measurable,
    {
        type Item = M;
        fn size() -> usize {
            MAX_BYTES
        }
    }

    /// Internal small string for `NodeText`.
    #[derive(Clone, Default)]
    #[repr(C)]
    pub struct LeafSmallVec<M>
    where
        M: Measurable,
    {
        buffer: SmallVec<BackingArray<M>>,
    }

    impl<M> LeafSmallVec<M>
    where
        M: Measurable,
    {
        #[inline(always)]
        pub fn new() -> Self {
            LeafSmallVec {
                buffer: SmallVec::new(),
            }
        }

        #[inline(always)]
        pub fn with_capacity(capacity: usize) -> Self {
            LeafSmallVec {
                buffer: SmallVec::with_capacity(capacity),
            }
        }

        #[inline(always)]
        pub fn from_slice(slice: &[M]) -> Self {
            let mut leaf_small_vec = LeafSmallVec::with_capacity(slice.len());
            leaf_small_vec.insert_slice(0, slice);
            leaf_small_vec
        }

        #[inline(always)]
        pub fn len(&self) -> usize {
            self.buffer.len()
        }

        pub fn as_slice(&self) -> &[M] {
            &self.buffer
        }

        /// Inserts a [`&[Measurable]`][Measurable] at [`byte_idx`][usize].
        ///
        /// Panics on out-of-bounds or of [`byte_idx`][usize] isn't a char boundary.
        #[inline(always)]
        pub fn insert_slice(&mut self, byte_idx: usize, slice: &[M]) {
            // Copy bytes from `slice` into the appropriate space in the
            // buffer.
            self.buffer.insert_from_slice(byte_idx, slice);
        }

        /// Removes text in range `[start_byte_idx, end_byte_idx)`
        ///
        /// Panics on out-of-bounds or non-char-boundary indices.
        #[inline(always)]
        pub fn remove_range(&mut self, start_byte_idx: usize, end_byte_idx: usize) {
            assert!(start_byte_idx <= end_byte_idx);
            // Already checked by copy_within/is_char_boundary.
            debug_assert!(end_byte_idx <= self.len());
            let len = self.len();
            let amt = end_byte_idx - start_byte_idx;

            self.buffer.copy_within(end_byte_idx..len, start_byte_idx);

            self.buffer.truncate(len - amt);
        }

        /// Removes text after `byte_idx`.
        #[inline(always)]
        pub fn truncate(&mut self, byte_idx: usize) {
            // Already checked by is_char_boundary.
            debug_assert!(byte_idx <= self.len());
            self.buffer.truncate(byte_idx);
        }

        /// Splits at `byte_idx`, returning the right part and leaving the
        /// left part in the original.
        ///
        /// Panics on out-of-bounds or of `byte_idx` isn't a char boundary.
        #[inline(always)]
        pub fn split_off(&mut self, byte_idx: usize) -> Self {
            // Already checked by is_char_boundary.
            debug_assert!(byte_idx <= self.len());
            let len = self.len();
            let mut other = LeafSmallVec::with_capacity(len - byte_idx);
            other.buffer.extend_from_slice(&self.buffer[byte_idx..]);
            self.buffer.truncate(byte_idx);
            other
        }

        /// Re-inlines the data if it's been heap allocated but can
        /// fit inline.
        #[inline(always)]
        pub fn inline_if_possible(&mut self) {
            if self.buffer.spilled() && (self.buffer.len() <= self.buffer.inline_size()) {
                self.buffer.shrink_to_fit();
            }
        }
    }

    //-----------------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn small_string_basics() {
            let s = LeafSmallVec::from_str("Hello!");
            assert_eq!("Hello!", s.as_str());
            assert_eq!(6, s.len());
        }

        #[test]
        fn insert_str_01() {
            let mut s = LeafSmallVec::from_str("Hello!");
            s.insert_str(3, "oz");
            assert_eq!("Helozlo!", s.as_str());
        }

        #[test]
        #[should_panic]
        fn insert_str_02() {
            let mut s = LeafSmallVec::from_str("Hello!");
            s.insert_str(7, "oz");
        }

        #[test]
        #[should_panic]
        fn insert_str_03() {
            let mut s = LeafSmallVec::from_str("こんにちは");
            s.insert_str(4, "oz");
        }

        #[test]
        fn remove_range_01() {
            let mut s = LeafSmallVec::from_str("Hello!");
            s.remove_range(2, 4);
            assert_eq!("Heo!", s.as_str());
        }

        #[test]
        #[should_panic]
        fn remove_range_02() {
            let mut s = LeafSmallVec::from_str("Hello!");
            s.remove_range(4, 2);
        }

        #[test]
        #[should_panic]
        fn remove_range_03() {
            let mut s = LeafSmallVec::from_str("Hello!");
            s.remove_range(2, 7);
        }

        #[test]
        #[should_panic]
        fn remove_range_04() {
            let mut s = LeafSmallVec::from_str("こんにちは");
            s.remove_range(2, 4);
        }

        #[test]
        fn truncate_01() {
            let mut s = LeafSmallVec::from_str("Hello!");
            s.truncate(4);
            assert_eq!("Hell", s.as_str());
        }

        #[test]
        #[should_panic]
        fn truncate_02() {
            let mut s = LeafSmallVec::from_str("Hello!");
            s.truncate(7);
        }

        #[test]
        #[should_panic]
        fn truncate_03() {
            let mut s = LeafSmallVec::from_str("こんにちは");
            s.truncate(4);
        }

        #[test]
        fn split_off_01() {
            let mut s1 = LeafSmallVec::from_str("Hello!");
            let s2 = s1.split_off(4);
            assert_eq!("Hell", s1.as_str());
            assert_eq!("o!", s2.as_str());
        }

        #[test]
        #[should_panic]
        fn split_off_02() {
            let mut s1 = LeafSmallVec::from_str("Hello!");
            s1.split_off(7);
        }

        #[test]
        #[should_panic]
        fn split_off_03() {
            let mut s1 = LeafSmallVec::from_str("こんにちは");
            s1.split_off(4);
        }
    }
}
