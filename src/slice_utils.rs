use crate::rope::Measurable;

pub fn width_of<M>(slice: &[M]) -> usize
where
    M: Measurable,
{
    slice.iter().map(|measurable| measurable.width()).sum()
}

/// Gets the width sum up to a given `index` in the `slice`.
pub fn index_to_width<M>(slice: &[M], index: usize) -> usize
where
    M: Measurable,
{
    slice
        .iter()
        .take(index)
        .map(|measurable| measurable.width())
        .sum()
}

/// Finds the index of the element whose starting width sum matches `width`.
pub fn start_width_to_index<M>(slice: &[M], width: usize) -> usize
where
    M: Measurable,
{
    let mut index = 0;
    let mut accum = 0;

    for measurable in slice {
        let measurable_width = measurable.width();
        let new_accum = accum + measurable_width;

        if (measurable_width == 0 && new_accum >= width) || new_accum > width {
            break;
        }
        accum = new_accum;
        index += 1;
    }

    index
}

/// Finds the index of the element whose ending width sum matches `width`.
pub fn end_width_to_index<M>(slice: &[M], width: usize) -> usize
where
    M: Measurable,
{
    let mut index = 0;
    let mut accum = 0;

    for measurable in slice {
        let measurable_width = measurable.width();
        // This makes it so that every 0 width node exactly at `width` is also captured.
        if accum > width || measurable_width != 0 && accum == width {
            break;
        }

        accum += measurable_width;
        index += 1;
    }

    index
}
