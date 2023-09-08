use std::ops::{Add, AddAssign, Sub, SubAssign};

use crate::{slice_utils::measure_of, tree::Count, Measurable};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct SliceInfo<M>
where
    M: Measurable,
{
    pub(crate) len: Count,
    pub(crate) measure: M::Measure,
}

impl<M> SliceInfo<M>
where
    M: Measurable,
{
    #[inline]
    pub fn new() -> Self {
        Self {
            len: 0,
            measure: M::Measure::default(),
        }
    }

    #[inline]
    pub fn from_slice(slice: &[M]) -> Self {
        Self {
            len: slice.len() as Count,
            measure: measure_of(slice),
        }
    }
}

impl<M> Add for SliceInfo<M>
where
    M: Measurable,
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

impl<M> AddAssign for SliceInfo<M>
where
    M: Measurable,
{
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<M> Sub for SliceInfo<M>
where
    M: Measurable,
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

impl<M> SubAssign for SliceInfo<M>
where
    M: Measurable,
{
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}
