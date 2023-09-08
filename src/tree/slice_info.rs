use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Sub, SubAssign},
};

use crate::{slice_utils::measure_of, tree::Count, Measurable};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct SliceInfo<T>
where
    T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
{
    pub(crate) len: Count,
    pub(crate) measure: T,
}

impl<T> SliceInfo<T>
where
    T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
{
    #[inline]
    pub fn new<M: Measurable>() -> SliceInfo<M::Measure> {
        SliceInfo {
            len: 0,
            measure: M::Measure::default(),
        }
    }

    #[inline]
    pub fn from_slice<M: Measurable>(slice: &[M]) -> SliceInfo<M::Measure> {
        SliceInfo {
            len: slice.len() as Count,
            measure: measure_of(slice),
        }
    }
}

impl<T> Add for SliceInfo<T>
where
    T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            len: self.len + rhs.len,
            measure: self.measure + rhs.measure,
        }
    }
}

impl<T> AddAssign for SliceInfo<T>
where
    T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
{
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<T> Sub for SliceInfo<T>
where
    T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            len: self.len - rhs.len,
            measure: self.measure - rhs.measure,
        }
    }
}

impl<T> SubAssign for SliceInfo<T>
where
    T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
{
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}
