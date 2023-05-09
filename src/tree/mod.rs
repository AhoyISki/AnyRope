mod branch_children;
mod leaf_slice;
mod node;
mod slice_info;

pub(crate) use self::branch_children::BranchChildren;
pub(crate) use self::leaf_slice::LeafSlice;
pub(crate) use self::node::Node;
pub(crate) use self::slice_info::SliceInfo;

// Type used for storing tree metadata, such as indices and widths.
pub(crate) type Count = u64;

// Real constants used in release builds.
#[cfg(not(any(test, feature = "small_chunks")))]
mod constants {
    use super::SliceInfo;
    use smallvec::SmallVec;
    use std::{
        mem::{align_of, size_of},
        sync::Arc,
    };

    // Because stdlib's max is not const for some reason.
    // TODO: replace with stdlib max once it's const.
    const fn cmax(a: usize, b: usize) -> usize {
        if a > b {
            a
        } else {
            b
        }
    }

    // Aim for Node + Arc counters to be 1024 bytes.  Keeping the nodes
    // multiples of large powers of two makes it easier for the memory
    // allocator to avoid fragmentation.
    const TARGET_TOTAL_SIZE: usize = 1024;

    // Space that the strong and weak Arc counters take up in `ArcInner`.
    const ARC_COUNTERS_SIZE: usize = size_of::<std::sync::atomic::AtomicUsize>() * 2;

    // Misc useful info that we need below.
    const NODE_CHILDREN_ALIGN: usize = cmax(align_of::<Arc<u8>>(), align_of::<SliceInfo>());
    const fn node_align<T>() -> usize {
        align_of::<SmallVec<[T; 16]>>()
    }
    const fn start_offset<T>() -> usize {
        let node_inner_align = cmax(NODE_CHILDREN_ALIGN, node_align::<T>());
        // The +NODE_INNER_ALIGN is because of Node's enum discriminant.
        ARC_COUNTERS_SIZE + node_inner_align
    }

    // Node maximums.
    pub const fn max_children<T>() -> usize {
        let node_list_align = align_of::<Arc<u8>>();
        let info_list_align = align_of::<SliceInfo>();
        let field_gap = if node_list_align >= info_list_align {
            0
        } else {
            // This is over-conservative, because in reality it depends
            // on the number of elements.  But handling that is probably
            // more complexity than it's worth.
            info_list_align - node_list_align
        };

        // The -NODE_CHILDREN_ALIGN is for the `len` field in `NodeChildrenInternal`.
        let target_size = TARGET_TOTAL_SIZE - start_offset::<T>() - NODE_CHILDREN_ALIGN - field_gap;

        target_size / (size_of::<Arc<u8>>() + size_of::<SliceInfo>())
    }
    pub const fn max_len<T>() -> usize {
        let smallvec_overhead = size_of::<SmallVec<[T; 16]>>() - 16;
        TARGET_TOTAL_SIZE - start_offset::<T>() - smallvec_overhead
    }

    // Node minimums.
    // Note: min_len is intentionally a little smaller than half
    // max_len, to give a little wiggle room when on the edge of
    // merging/splitting.
    pub const fn min_children<T>() -> usize {
        max_children::<T>() / 2
    }
    pub const fn min_len<T>() -> usize {
        (max_len::<T>() / 2) - (max_len::<T>() / 32)
    }
}

// Smaller constants used in debug builds. These are different from release
// in order to trigger deeper trees without having to use huge slice data in
// the tests.
#[cfg(any(test))]
mod test_constants {
    pub const fn max_children<T>() -> usize {
        5
    }
    pub const fn min_children<T>() -> usize {
        max_children::<T>() / 2
    }

    pub const fn max_len<T>() -> usize {
        9
    }
    pub const fn min_len<T>() -> usize {
        (max_len::<T>() / 2) - (max_len::<T>() / 32)
    }
}

#[cfg(not(test))]
pub use self::constants::{max_children, max_len, min_children, min_len};

#[cfg(test)]
pub use self::test_constants::{max_children, max_len, min_children, min_len};
