use std::sync::Arc;

use smallvec::SmallVec;

use crate::rope::{Measurable, Rope};
use crate::tree::{Branch, Leaf, Node, MAX_BYTES, MAX_CHILDREN, MIN_BYTES};

/// An efficient incremental `Rope` builder.
///
/// This is used to efficiently build ropes from sequences of text
/// chunks.  It is useful for creating ropes from:
///
/// - ...large text files, without pre-loading their entire contents into
///   memory (but see [`from_reader()`](Rope::from_reader) for a convenience
///   function that does this for casual use-cases).
/// - ...streaming data sources.
/// - ...non-utf8 text data, doing the encoding conversion incrementally
///   as you go.
///
/// Unlike repeatedly calling `Rope::insert()` on the end of a rope,
/// this API runs in time linear to the amount of data fed to it, and
/// is overall much faster.
///
/// # Example
/// ```
/// # use ropey::RopeBuilder;
/// #
/// let mut builder = RopeBuilder::new();
///
/// builder.append("Hello ");
/// builder.append("world!\n");
/// builder.append("How's ");
/// builder.append("it goin");
/// builder.append("g?");
///
/// let rope = builder.finish();
///
/// assert_eq!(rope, "Hello world!\nHow's it going?");
/// ```
#[derive(Debug, Clone)]
pub struct RopeBuilder<M>
where
    M: Measurable,
{
    stack: SmallVec<[Arc<Node<M>>; 4]>,
    buffer: Vec<M>,
    last_chunk_len_bytes: usize,
}

impl<M> RopeBuilder<M>
where
    M: Measurable,
{
    /// Creates a new RopeBuilder, ready for input.
    pub fn new() -> Self {
        RopeBuilder {
            stack: {
                let mut stack = SmallVec::new();
                stack.push(Arc::new(Node::new()));
                stack
            },
            buffer: Vec::new(),
            last_chunk_len_bytes: 0,
        }
    }

    /// Appends `chunk` to the end of the in-progress `Rope`.
    ///
    /// Call this method repeatedly to incrementally build up a
    /// `Rope`.  The passed text chunk can be as large or small as
    /// desired, but larger chunks are more efficient.
    ///
    /// `chunk` must be valid utf8 text.
    pub fn append(&mut self, chunk: &[M]) {
        self.append_internal(chunk, false);
    }

    /// Finishes the build, and returns the `Rope`.
    ///
    /// Note: this method consumes the builder.  If you want to continue
    /// building other ropes with the same prefix, you can clone the builder
    /// before calling `finish()`.
    pub fn finish(mut self) -> Rope<M> {
        // Append the last leaf
        self.append_internal(&[], true);
        self.finish_internal(true)
    }

    /// Builds a rope all at once from a single string slice.
    ///
    /// This avoids the creation and use of the internal buffer.  This is
    /// for internal use only, because the public-facing API has
    /// Rope::from_str(), which actually uses this for its implementation.
    pub(crate) fn build_at_once(mut self, chunk: &[M]) -> Rope<M> {
        self.append_internal(chunk, true);
        self.finish_internal(true)
    }

    /// NOT PART OF THE PUBLIC API (hidden from docs for a reason!).
    ///
    /// Appends `contents` to the in-progress rope as a single leaf
    /// node (chunk).  This is useful for building ropes with specific
    /// chunk configurations for testing purposes.  It will happily append
    /// both empty and more-than-max-size chunks.
    ///
    /// This makes no attempt to be consistent with the standard `append()`
    /// method, and should not be used in conjunction with it.
    #[doc(hidden)]
    pub fn _append_chunk(&mut self, contents: &[M]) {
        self.append_leaf_node(Arc::new(Node::Leaf(Leaf::from(contents))));
    }

    /// NOT PART OF THE PUBLIC API (hidden from docs for a reason!).
    ///
    /// Finishes the build without doing any tree fixing to adhere
    /// to the btree invariants. To be used with `_append_chunk()` to
    /// construct ropes with specific chunk boundaries for testing.
    #[doc(hidden)]
    pub fn _finish_no_fix(self) -> Rope<M> {
        self.finish_internal(false)
    }

    //-----------------------------------------------------------------

    // Internal workings of `append()`.
    fn append_internal(&mut self, chunk: &[M], is_last_chunk: bool) {
        let mut chunk = chunk;

        // Repeatedly chop text off the end of the input, creating
        // leaf nodes out of them and appending them to the tree.
        while !chunk.is_empty() || (!self.buffer.is_empty() && is_last_chunk) {
            // Get the text for the next leaf
            let (leaf_text, remainder) = self.get_next_leaf_slice(chunk, is_last_chunk);
            chunk = remainder;

            self.last_chunk_len_bytes = chunk.len();

            // Append the leaf to the rope
            match leaf_text {
                NextSlice::None => break,
                NextSlice::UseBuffer => {
                    let leaf_text = Leaf::from(self.buffer.as_slice());
                    self.append_leaf_node(Arc::new(Node::Leaf(leaf_text)));
                    self.buffer.clear();
                }
                NextSlice::Slice(s) => {
                    self.append_leaf_node(Arc::new(Node::Leaf(Leaf::from(s))));
                }
            }
        }
    }

    // Internal workings of `finish()`.
    //
    // When `fix_tree` is false, the resulting node tree is NOT fixed up
    // to adhere to the btree invariants.  This is useful for some testing
    // code.  But generally, `fix_tree` should be set to true.
    fn finish_internal(mut self, fix_tree: bool) -> Rope<M> {
        // Zip up all the remaining nodes on the stack
        let mut stack_idx = self.stack.len() - 1;
        while stack_idx >= 1 {
            let node = self.stack.pop().unwrap();
            if let Node::Branch(ref mut children) = *Arc::make_mut(&mut self.stack[stack_idx - 1]) {
                children.push((node.slice_info(), node));
            } else {
                unreachable!();
            }
            stack_idx -= 1;
        }

        // Create the rope.
        let mut rope = Rope {
            root: self.stack.pop().unwrap(),
        };

        // Fix up the tree to be well-formed.
        if fix_tree {
            Arc::make_mut(&mut rope.root).zip_fix_right();
            if self.last_chunk_len_bytes < MIN_BYTES
                && self.last_chunk_len_bytes != rope.total_len()
            {
                // Merge the last chunk if it was too small.
                let idx = rope.total_width()
                    - rope.idx_to_width(rope.total_len() - self.last_chunk_len_bytes);
                Arc::make_mut(&mut rope.root).fix_tree_seam(idx);
            }
            rope.pull_up_singular_nodes();
        }

        return rope;
    }

    // Returns (next_leaf_text, remaining_text)
    #[inline(always)]
    fn get_next_leaf_slice<'a>(
        &mut self,
        slice: &'a [M],
        is_last_chunk: bool,
    ) -> (NextSlice<'a, M>, &'a [M]) {
        assert!(
            self.buffer.len() < MAX_BYTES,
            "RopeBuilder: buffer is already full when receiving a chunk! \
             This should never happen!",
        );

        // Simplest case: empty buffer and enough in `text` for a full
        // chunk, so just chop a chunk off from `text` and use that.
        if self.buffer.is_empty() && slice.len() >= MAX_BYTES {
            let split_idx = MAX_BYTES.min(slice.len() - 1);
            return (NextSlice::Slice(&slice[..split_idx]), &slice[split_idx..]);
        }
        // If the buffer + `text` is enough for a full chunk, push enough
        // of `text` onto the buffer to fill it and use that.
        else if (slice.len() + self.buffer.len()) >= MAX_BYTES {
            let mut split_idx = MAX_BYTES - self.buffer.len();
            self.buffer.extend_from_slice(&slice[..split_idx]);
            return (NextSlice::UseBuffer, &slice[split_idx..]);
        }
        // If we don't have enough text for a full chunk.
        else {
            // If it's our last chunk, wrap it all up!
            if is_last_chunk {
                if self.buffer.is_empty() {
                    return if slice.is_empty() {
                        (NextSlice::None, &[])
                    } else {
                        (NextSlice::Slice(slice), &[])
                    };
                } else {
                    self.buffer.extend_from_slice(slice);
                    return (NextSlice::UseBuffer, &[]);
                }
            }
            // Otherwise, just push to the buffer.
            else {
                self.buffer.extend_from_slice(slice);
                return (NextSlice::None, &[]);
            }
        }
    }

    fn append_leaf_node(&mut self, leaf: Arc<Node<M>>) {
        let last = self.stack.pop().unwrap();
        match *last {
            Node::Leaf(_) => {
                if last.leaf_slice().is_empty() {
                    self.stack.push(leaf);
                } else {
                    let mut children = Branch::new();
                    children.push((last.slice_info(), last));
                    children.push((leaf.slice_info(), leaf));
                    self.stack.push(Arc::new(Node::Branch(children)));
                }
            }

            Node::Branch(_) => {
                self.stack.push(last);
                let mut left = leaf;
                let mut stack_idx = (self.stack.len() - 1) as isize;
                loop {
                    if stack_idx < 0 {
                        // We're above the root, so do a root split.
                        let mut children = Branch::new();
                        children.push((left.slice_info(), left));
                        self.stack.insert(0, Arc::new(Node::Branch(children)));
                        break;
                    } else if self.stack[stack_idx as usize].child_count() < (MAX_CHILDREN - 1) {
                        // There's room to add a child, so do that.
                        Arc::make_mut(&mut self.stack[stack_idx as usize])
                            .children_mut()
                            .push((left.slice_info(), left));
                        break;
                    } else {
                        // Not enough room to fit a child, so split.
                        left = Arc::new(Node::Branch(
                            Arc::make_mut(&mut self.stack[stack_idx as usize])
                                .children_mut()
                                .push_split((left.slice_info(), left)),
                        ));
                        std::mem::swap(&mut left, &mut self.stack[stack_idx as usize]);
                        stack_idx -= 1;
                    }
                }
            }
        }
    }
}

impl<M> Default for RopeBuilder<M>
where
    M: Measurable,
{
    fn default() -> Self {
        Self::new()
    }
}

enum NextSlice<'a, M>
where
    M: Measurable,
{
    None,
    UseBuffer,
    Slice(&'a [M]),
}

//===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // 127 bytes, 103 chars, 4 lines
    const TEXT: &str = "Hello there!  How're you doing?\r\nIt's \
                        a fine day, isn't it?\r\nAren't you glad \
                        we're alive?\r\nこんにちは、みんなさん！";

    #[test]
    fn rope_builder_01() {
        let mut b = RopeBuilder::new();

        b.append("Hello there!  How're you doing?\r");
        b.append("\nIt's a fine ");
        b.append("d");
        b.append("a");
        b.append("y,");
        b.append(" ");
        b.append("isn't it?");
        b.append("\r");
        b.append("\nAren't you ");
        b.append("glad we're alive?\r");
        b.append("\n");
        b.append("こんにち");
        b.append("は、みんなさ");
        b.append("ん！");

        let r = b.finish();

        assert_eq!(r, TEXT);

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn rope_builder_default_01() {
        let mut b = RopeBuilder::default();

        b.append("Hello there!  How're you doing?\r");
        b.append("\nIt's a fine day, isn't it?\r\nAren't you ");
        b.append("glad we're alive?\r\nこんにちは、みんなさん！");

        let r = b.finish();

        assert_eq!(r, TEXT);

        r.assert_integrity();
        r.assert_invariants();
    }
}
