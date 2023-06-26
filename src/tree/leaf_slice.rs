use std::borrow::Borrow;
use std::fmt::{Debug, Display};
use std::ops::Deref;

use crate::max_children;
use crate::rope::Measurable;

use self::inner::LeafSmallVec;

use super::max_len;

/// A custom small string.  The unsafe guts of this are in [`LeafSmallVec`]
/// further down in this file.
#[derive(Clone, Default)]
#[repr(C)]
pub(crate) struct LeafSlice<M>(inner::LeafSmallVec<M>)
where
    M: Measurable,
    [(); max_len::<M>()]: Sized;

impl<M> LeafSlice<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    /// Creates a new [`Leaf`] from a slice.
    pub fn from_slice(value: &[M]) -> Self {
        Self(LeafSmallVec::from_slice(value))
    }

    pub fn insert_slice(&mut self, index: usize, slice: &[M]) {
        self.0.insert_slice(index, slice);
    }

    /// Inserts a [`&[M]`][Measurable] at `index` and splits the resulting
    /// string in half, returning the right half.
    pub fn insert_slice_split(&mut self, split_index: usize, slice: &[M]) -> Self {
        let tot_len = self.len() + slice.len();
        let mid_index = tot_len / 2;

        self.0.insert_slice(split_index, slice);
        let right = LeafSlice::from_slice(&self[mid_index..]);
        self.truncate(mid_index);

        self.0.inline_if_possible();
        right
    }

    /// Appends a [`&[M]`][Measurable] to end the of the [`LeafSlice`].
    pub fn push_slice(&mut self, slice: &[M]) {
        let len = self.len();
        self.0.insert_slice(len, slice);
    }

    /// Appends a [`&[M]`][Measurable] and splits the resulting string in half,
    /// returning the right half.
    ///
    /// Only splits on code point boundaries and will never split CRLF pairs,
    /// so if the whole string is a single code point or CRLF pair, the split
    /// will fail and the returned string will be empty.
    pub fn push_slice_split(&mut self, slice: &[M]) -> Self {
        let len = self.len();
        self.insert_slice_split(len, slice)
    }

    /// Drops the text after `index`.
    pub fn truncate(&mut self, index: usize) {
        self.0.truncate(index);
        self.0.inline_if_possible();
    }

    /// Drops the text before `index`, shifting the
    /// rest of the text to fill in the space.
    pub fn truncate_front(&mut self, index: usize) {
        self.0.remove_range(0, index);
        self.0.inline_if_possible();
    }

    /// Removes the text in the index interval `[start_index, end_index)`.
    pub fn remove_range(&mut self, start_index: usize, end_index: usize) {
        self.0.remove_range(start_index, end_index);
        self.0.inline_if_possible();
    }

    /// Splits the [`LeafSlice`] at `index`.
    ///
    /// The left part remains in the original, and the right part is
    /// returned in a new [`LeafSlice`].
    pub fn split_off(&mut self, index: usize) -> Self {
        let other = LeafSlice(self.0.split_off(index));
        self.0.inline_if_possible();
        other
    }

    pub fn zero_width_end(&self) -> bool {
        self.0
            .as_slice()
            .last()
            .map(|measurable| measurable.width() == 0)
            .unwrap_or(false)
    }
}

impl<M> std::cmp::PartialEq for LeafSlice<M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M>()]: Sized,
{
    fn eq(&self, other: &Self) -> bool {
        let (s1, s2): (&[M], &[M]) = (self, other);
        s1 == s2
    }
}

impl<'a, M> PartialEq<LeafSlice<M>> for &'a [M]
where
    M: Measurable + PartialEq,
    [(); max_len::<M>()]: Sized,
{
    fn eq(&self, other: &LeafSlice<M>) -> bool {
        *self == (other as &[M])
    }
}

impl<'a, M> PartialEq<&'a [M]> for LeafSlice<M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M>()]: Sized,
{
    fn eq(&self, other: &&'a [M]) -> bool {
        (self as &[M]) == *other
    }
}

impl<M> std::fmt::Display for LeafSlice<M>
where
    M: Measurable + Display + Debug,
    [(); max_len::<M>()]: Sized,
{
    fn fmt(&self, fm: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        LeafSlice::deref(self).fmt(fm)
    }
}

impl<M> std::fmt::Debug for LeafSlice<M>
where
    M: Measurable + Debug,
    [(); max_len::<M>()]: Sized,
{
    fn fmt(&self, fm: &mut std::fmt::Formatter) -> std::fmt::Result {
        LeafSlice::deref(self).fmt(fm)
    }
}

impl<M> Deref for LeafSlice<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
{
    type Target = [M];

    fn deref(&self) -> &[M] {
        self.0.as_slice()
    }
}

impl<M> AsRef<[M]> for LeafSlice<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
{
    fn as_ref(&self) -> &[M] {
        self.0.as_slice()
    }
}

impl<M> Borrow<[M]> for LeafSlice<M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
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
    use crate::tree::max_len;
    use smallvec::{Array, SmallVec};

    use super::Measurable;

    /// The backing internal buffer type for [`LeafSlice`][super::LeafSlice].
    #[derive(Copy, Clone)]
    struct BackingArray<M>([M; max_len::<M>()])
    where
        M: Measurable,
        [(); max_len::<M>()]: Sized;

    /// We need a very specific size of array, which is not necessarily
    /// supported directly by the impls in the smallvec crate.  We therefore
    /// have to implement this unsafe trait for our specific array size.
    /// TODO: once integer const generics land, and smallvec updates its APIs
    /// to use them, switch over and get rid of this unsafe impl.
    unsafe impl<M> Array for BackingArray<M>
    where
        M: Measurable,
        [(); max_len::<M>()]: Sized,
    {
        type Item = M;
        fn size() -> usize {
            max_len::<M>()
        }
    }

    /// Internal small string for [`LeafSlice`][super::LeafSlice].
    #[derive(Clone, Default)]
    #[repr(C)]
    pub struct LeafSmallVec<M>
    where
        M: Measurable,
        [(); max_len::<M>()]: Sized,
    {
        buffer: SmallVec<BackingArray<M>>,
    }

    impl<M> LeafSmallVec<M>
    where
        M: Measurable,
        [(); max_len::<M>()]: Sized,
    {
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

        /// Inserts a [`&[M]`][Measurable] at `index`.
        #[inline(always)]
        pub fn insert_slice(&mut self, index: usize, slice: &[M]) {
            // Copy the `slice` into the appropriate space in the buffer.
            self.buffer.insert_from_slice(index, slice);
        }

        /// Removes text in range `[start_index, end_index)`
        #[inline(always)]
        pub fn remove_range(&mut self, start_index: usize, end_index: usize) {
            assert!(start_index <= end_index);
            // Already checked by copy_within/is_char_boundary.
            debug_assert!(end_index <= self.len());
            let len = self.len();
            let amt = end_index - start_index;

            self.buffer.copy_within(end_index..len, start_index);

            self.buffer.truncate(len - amt);
        }

        /// Removes text after `index`.
        #[inline(always)]
        pub fn truncate(&mut self, index: usize) {
            // Already checked by is_char_boundary.
            debug_assert!(index <= self.len());
            self.buffer.truncate(index);
        }

        /// Splits at `index`, returning the right part and leaving the
        /// left part in the original.
        #[inline(always)]
        pub fn split_off(&mut self, index: usize) -> Self {
            debug_assert!(index <= self.len());
            let len = self.len();
            let mut other = LeafSmallVec::with_capacity(len - index);
            other.buffer.extend_from_slice(&self.buffer[index..]);
            self.buffer.truncate(index);
            other
        }

        /// Re-inlines the data if it's been heap allocated but can fit inline.
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
        use crate::Width;

        #[test]
        fn vec_basics() {
            let vec = LeafSmallVec::from_slice(&[Width(1), Width(2), Width(3), Width(0), Width(0)]);
            assert_eq!(vec.as_slice(), &[
                Width(1),
                Width(2),
                Width(3),
                Width(0),
                Width(0)
            ]);
            assert_eq!(5, vec.len());
        }

        #[test]
        fn insert_slice_01() {
            let mut vec = LeafSmallVec::from_slice(&[Width(1), Width(2)]);
            vec.insert_slice(2, &[Width(5), Width(0), Width(0)]);
            assert_eq!(vec.as_slice(), &[
                Width(1),
                Width(2),
                Width(5),
                Width(0),
                Width(0)
            ]);
        }

        #[test]
        #[should_panic]
        fn insert_slice_02() {
            let mut vec = LeafSmallVec::from_slice(&[Width(2), Width(0)]);
            vec.insert_slice(3, &[Width(1)]);
        }

        #[test]
        fn remove_range_01() {
            let mut vec =
                LeafSmallVec::from_slice(&[Width(1), Width(2), Width(1), Width(0), Width(0)]);
            vec.remove_range(2, 3);
            assert_eq!(vec.as_slice(), &[Width(1), Width(2), Width(0), Width(0)]);
        }

        #[test]
        #[should_panic]
        fn remove_range_02() {
            let mut vec = LeafSmallVec::from_slice(&[Width(5), Width(0), Width(0), Width(2)]);
            vec.remove_range(4, 2);
        }

        #[test]
        #[should_panic]
        fn remove_range_03() {
            let mut vec = LeafSmallVec::from_slice(&[Width(5), Width(0), Width(0), Width(2)]);
            vec.remove_range(2, 7);
        }

        #[test]
        fn truncate_01() {
            let mut vec = LeafSmallVec::from_slice(&[Width(3), Width(0), Width(0), Width(4)]);
            vec.truncate(3);
            assert_eq!(vec.as_slice(), &[Width(3), Width(0), Width(0)]);
        }

        #[test]
        #[should_panic]
        fn truncate_02() {
            let mut vec = LeafSmallVec::from_slice(&[Width(6)]);
            vec.truncate(7);
        }

        #[test]
        fn split_off_01() {
            let mut vec_1 = LeafSmallVec::from_slice(&[Width(1), Width(3), Width(0), Width(0)]);
            let vec_2 = vec_1.split_off(2);
            assert_eq!(vec_1.as_slice(), &[Width(1), Width(3)]);
            assert_eq!(vec_2.as_slice(), &[Width(0), Width(0)]);
        }

        #[test]
        #[should_panic]
        fn split_off_02() {
            let mut s1 =
                LeafSmallVec::from_slice(&[Width(1), Width(2), Width(3), Width(0), Width(0)]);
            s1.split_off(7);
        }
    }
}
