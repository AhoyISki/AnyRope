use std::ops::{Add, AddAssign, Sub, SubAssign};

use crate::{rope::Measurable, slice_utils::width_of, tree::Count};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct SliceInfo {
    pub(crate) len: Count,
    pub(crate) width: Count,
}

impl SliceInfo {
    #[inline]
    pub fn new() -> SliceInfo {
        SliceInfo { len: 0, width: 0 }
    }

    #[inline]
    pub fn from_slice<M>(slice: &[M]) -> SliceInfo
    where
        M: Measurable,
    {
        SliceInfo {
            len: slice.len() as Count,
            width: width_of(slice) as Count,
        }
    }
}

impl Add for SliceInfo {
    type Output = Self;

    #[inline]
    fn add(self, rhs: SliceInfo) -> SliceInfo {
        SliceInfo {
            len: self.len + rhs.len,
            width: self.width + rhs.width,
        }
    }
}

impl AddAssign for SliceInfo {
    #[inline]
    fn add_assign(&mut self, other: SliceInfo) {
        *self = *self + other;
    }
}

impl Sub for SliceInfo {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: SliceInfo) -> SliceInfo {
        SliceInfo {
            len: self.len - rhs.len,
            width: self.width - rhs.width,
        }
    }
}

impl SubAssign for SliceInfo {
    #[inline]
    fn sub_assign(&mut self, other: SliceInfo) {
        *self = *self - other;
    }
}
