mod branch_children;
mod leaf_slice;
mod node;
mod slice_info;

pub(crate) use self::{
    branch_children::BranchChildren, leaf_slice::LeafSlice, node::Node, slice_info::SliceInfo,
};

// Type used for storing tree metadata, such as indices and widths.
pub(crate) type Count = u64;

// Real constants used in release builds.
#[cfg(not(test))]
mod constants {
    use std::{
        fmt::Debug,
        mem::{align_of, size_of},
        ops::{Add, Sub},
        sync::Arc,
    };

    use smallvec::SmallVec;

    use super::SliceInfo;
    use crate::Measurable;

    // Because stdlib's max is not const for some reason.
    // TODO: replace with stdlib max once it's const.
    const fn cmax(a: usize, b: usize) -> usize {
        if a > b { a } else { b }
    }

    // Aim for Node + Arc counters to be 1024 bytes.  Keeping the nodes
    // multiples of large powers of two makes it easier for the memory
    // allocator to avoid fragmentation.
    const TARGET_TOTAL_SIZE: usize = 1024;

    // Space that the strong and weak Arc counters take up in `ArcInner`.
    const ARC_COUNTERS_SIZE: usize = size_of::<std::sync::atomic::AtomicUsize>() * 2;

    // Misc useful info that we need below.
    const fn node_children_align<T>() -> usize
    where
        T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
    {
        cmax(align_of::<Arc<u8>>(), align_of::<(SliceInfo<T>, bool)>())
    }

    const fn node_align<M>() -> usize {
        align_of::<SmallVec<[M; 16]>>()
    }

    const fn start_offset<M, T>() -> usize
    where
        T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
    {
        let node_inner_align = cmax(node_children_align::<T>(), node_align::<M>());
        // The `+ node_inner_align` is because of Node's enum discriminant.
        ARC_COUNTERS_SIZE + node_inner_align
    }

    // Node maximums.
    pub const fn max_children<M, T>() -> usize
    where
        T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
    {
        let node_list_align = align_of::<Arc<u8>>();
        let info_list_align = align_of::<SliceInfo<T>>();
        let field_gap = if node_list_align >= info_list_align {
            0
        } else {
            // This is over-conservative, because in reality it depends
            // on the number of elements.  But handling that is probably
            // more complexity than it's worth.
            info_list_align - node_list_align
        };

        // The -NODE_CHILDREN_ALIGN is for the `len` field in `NodeChildrenInternal`.
        let target_size =
            TARGET_TOTAL_SIZE - start_offset::<M, T>() - node_children_align::<T>() - field_gap;

        target_size / (size_of::<Arc<u8>>() + size_of::<SliceInfo<T>>())
    }
    pub const fn max_len<M, T>() -> usize
    where
        T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
    {
        let smallvec_overhead = size_of::<SmallVec<[M; 16]>>();
        (TARGET_TOTAL_SIZE - start_offset::<M, T>() - smallvec_overhead) / size_of::<M>()
    }

    // Node minimums.
    // Note: min_len is intentionally a little smaller than half
    // max_len, to give a little wiggle room when on the edge of
    // merging/splitting.
    pub const fn min_children<M, T>() -> usize
    where
        T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
    {
        max_children::<M, T>() / 2
    }
    pub const fn min_len<M, T>() -> usize
    where
        T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
    {
        (max_len::<M, T>() / 2) - (max_len::<M, T>() / 32)
    }
}

// Smaller constants used in debug builds. These are different from release
// in order to trigger deeper trees without having to use huge slice data in
// the tests.
#[cfg(test)]
mod test_constants {
    use std::{
        fmt::Debug,
        ops::{Add, Sub},
    };

    pub const fn max_children<M, T>() -> usize
    where
        T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
    {
        5
    }
    pub const fn min_children<M, T>() -> usize
    where
        T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
    {
        max_children::<M, T>() / 2
    }

    pub const fn max_len<M, T>() -> usize
    where
        T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
    {
        9
    }
    pub const fn min_len<M, T>() -> usize
    where
        T: Debug + Copy + Clone + PartialEq + Add<Output = T> + Sub<Output = T>,
    {
        (max_len::<M, T>() / 2) - (max_len::<M, T>() / 32)
    }
}

#[cfg(not(test))]
pub use self::constants::{max_children, max_len, min_children, min_len};
#[cfg(test)]
pub use self::test_constants::{max_children, max_len, min_children, min_len};
