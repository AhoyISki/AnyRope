use std::cmp::Ordering;

use crate::Measurable;

#[inline(always)]
pub fn measure_of<M: Measurable>(slice: &[M]) -> M::Measure {
    slice
        .iter()
        .map(|measurable| measurable.measure())
        .fold(M::Measure::default(), |accum, measure| accum + measure)
}

/// Gets the width sum up to a given `index` in the `slice`.
#[inline(always)]
pub fn index_to_measure<M: Measurable>(slice: &[M], index: usize) -> M::Measure {
    slice
        .iter()
        .take(index)
        .map(|measurable| measurable.measure())
        .fold(M::Measure::default(), |accum, measure| accum + measure)
}

/// Finds the index of the element whose starting width sum matches `width`.
#[inline(always)]
pub fn start_measure_to_index<M: Measurable>(
    slice: &[M],
    start: M::Measure,
    cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
) -> usize {
    let mut index = 0;
    let mut accum = M::Measure::default();

    for measurable in slice {
        let measure = measurable.measure();
        let next_accum = accum + measure;

        if cmp(&start, &next_accum).is_le()
            || (measure == M::Measure::default() && cmp(&start, &next_accum).is_eq())
        {
            break;
        }
        accum = next_accum;
        index += 1;
    }

    index
}

/// Finds the index of the element whose ending width sum matches `width`.
#[inline(always)]
pub fn end_measure_to_index<M: Measurable>(
    slice: &[M],
    end: M::Measure,
    cmp: impl Fn(&M::Measure, &M::Measure) -> Ordering,
) -> usize {
    let mut index = 0;
    let mut accum = M::Measure::default();

    for measurable in slice {
        let measure = measurable.measure();
        // This makes it so that every 0 width node exactly at `width` is also captured.
        if cmp(&end, &accum).is_le()
            || (measure != M::Measure::default() && cmp(&end, &accum).is_eq())
        {
            break;
        }

        accum += measure;
        index += 1;
    }

    index
}
