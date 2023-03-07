use std::fmt::Debug;
use std::iter::FromIterator;
use std::ops::RangeBounds;
use std::sync::Arc;

use crate::iter::{Chunks, Iter};
use crate::rope_builder::RopeBuilder;
use crate::slice::RopeSlice;
use crate::slice_utils::{idx_to_width, width_of, width_to_idx};
use crate::tree::{Branch, Node, SliceInfo, MAX_BYTES, MIN_BYTES};
use crate::{end_bound_to_num, start_bound_to_num, Error, Result};

pub trait Measurable: Clone + Copy {
    /// The width of this element, it need not be the actual lenght in bytes, but just a
    /// representative value, to be fed to the [Rope][crate::rope::Rope].
    fn width(&self) -> usize;
}

//impl<M> PartialEq<&'a [M]> for 

/// A utf8 text rope.
///
/// The time complexity of nearly all edit and query operations on `Rope` are
/// worst-case `O(log N)` in the length of the rope.  `Rope` is designed to
/// work efficiently even for huge (in the gigabytes) and pathological (all on
/// one line) texts.
///
/// # Editing Operations
///
/// The primary editing operations on `Rope` are insertion and removal of text.
/// For example:
///
/// ```
/// # use ropey::Rope;
/// #
/// let mut rope = Rope::from_str("Hello みんなさん!");
/// rope.remove(6..11);
/// rope.insert(6, "world");
///
/// assert_eq!(rope, "Hello world!");
/// ```
///
/// # Query Operations
///
/// `Rope` provides a rich set of efficient query functions, including querying
/// rope length in bytes/`char`s/lines, fetching individual `char`s or lines,
/// and converting between byte/`char`/line indices.  For example, to find the
/// starting `char` index of a given line:
///
/// ```
/// # use ropey::Rope;
/// #
/// let rope = Rope::from_str("Hello みんなさん!\nHow are you?\nThis text has multiple lines!");
///
/// assert_eq!(rope.line_to_char(0), 0);
/// assert_eq!(rope.line_to_char(1), 13);
/// assert_eq!(rope.line_to_char(2), 26);
/// ```
///
/// # Slicing
///
/// You can take immutable slices of a `Rope` using `slice()`:
///
/// ```
/// # use ropey::Rope;
/// #
/// let mut rope = Rope::from_str("Hello みんなさん!");
/// let middle = rope.slice(3..8);
///
/// assert_eq!(middle, "lo みん");
/// ```
///
/// # Cloning
///
/// Cloning `Rope`s is extremely cheap, running in `O(1)` time and taking a
/// small constant amount of memory for the new clone, regardless of text size.
/// This is accomplished by data sharing between `Rope` clones.  The memory
/// used by clones only grows incrementally as the their contents diverge due
/// to edits.  All of this is thread safe, so clones can be sent freely
/// between threads.
///
/// The primary intended use-case for this feature is to allow asynchronous
/// processing of `Rope`s.  For example, saving a large document to disk in a
/// separate thread while the user continues to perform edits.
#[derive(Clone)]
pub struct Rope<M>
where
    M: Measurable,
{
    pub(crate) root: Arc<Node<M>>,
}

impl<M> Rope<M>
where
    M: Measurable,
{
    //-----------------------------------------------------------------------
    // Constructors

    /// Creates an empty `Rope`.
    #[inline]
    pub fn new() -> Self {
        Rope {
            root: Arc::new(Node::new()),
        }
    }

    /// Creates a `Rope` from a string slice.
    ///
    /// Runs in O(N) time.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn from_slice(slice: &[M]) -> Self {
        RopeBuilder::new().build_at_once(slice)
    }

    //-----------------------------------------------------------------------
    // Informational methods

    /// Total number of bytes in the `Rope`.
    ///
    /// Runs in O(1) time.
    #[inline]
    pub fn total_len(&self) -> usize {
        self.root.count()
    }

    /// Total number of chars in the `Rope`.
    ///
    /// Runs in O(1) time.
    #[inline]
    pub fn total_width(&self) -> usize {
        self.root.width()
    }

    //-----------------------------------------------------------------------
    // Memory management methods

    /// Total size of the `Rope`'s text buffer space, in bytes.
    ///
    /// This includes unoccupied text buffer space.  You can calculate
    /// the unoccupied space with `capacity() - len_bytes()`.  In general,
    /// there will always be some unoccupied buffer space.
    ///
    /// Runs in O(N) time.
    pub fn capacity(&self) -> usize {
        let mut byte_count = 0;
        for chunk in self.chunks() {
            byte_count += chunk.len().max(MAX_BYTES);
        }
        byte_count
    }

    /// Shrinks the `Rope`'s capacity to the minimum possible.
    ///
    /// This will rarely result in `capacity() == len_bytes()`.  `Rope`
    /// stores text in a sequence of fixed-capacity chunks, so an exact fit
    /// only happens for texts that are both a precise multiple of that
    /// capacity _and_ have code point boundaries that line up exactly with
    /// the capacity boundaries.
    ///
    /// After calling this, the difference between `capacity()` and
    /// `len_bytes()` is typically under 1KB per megabyte of text in the
    /// `Rope`.
    ///
    /// **NOTE:** calling this on a `Rope` clone causes it to stop sharing
    /// all data with its other clones.  In such cases you will very likely
    /// be _increasing_ total memory usage despite shrinking the `Rope`'s
    /// capacity.
    ///
    /// Runs in O(N) time, and uses O(log N) additional space during
    /// shrinking.
    pub fn shrink_to_fit(&mut self) {
        let mut node_stack = Vec::new();
        let mut builder = RopeBuilder::new();

        node_stack.push(self.root.clone());
        *self = Rope::new();

        loop {
            if node_stack.is_empty() {
                break;
            }

            if node_stack.last().unwrap().is_leaf() {
                builder.append(node_stack.last().unwrap().leaf_slice());
            } else if node_stack.last().unwrap().child_count() == 0 {
                node_stack.pop();
            } else {
                let (_, next_node) = Arc::make_mut(node_stack.last_mut().unwrap())
                    .children_mut()
                    .remove(0);
                node_stack.push(next_node);
            }
        }

        *self = builder.finish();
    }

    //-----------------------------------------------------------------------
    // Edit methods

    /// Inserts `text` at char index `char_idx`.
    ///
    /// Runs in O(M + log N) time, where N is the length of the `Rope` and M
    /// is the length of `text`.
    ///
    /// # Panics
    ///
    /// Panics if `char_idx` is out of bounds (i.e. `char_idx > len_chars()`).
    #[inline]
    pub fn insert(&mut self, char_idx: usize, slice: &[M]) {
        // Bounds check
        self.try_insert(char_idx, slice).unwrap()
    }

    /// Inserts a single char `ch` at char index `char_idx`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_idx` is out of bounds (i.e. `char_idx > len_chars()`).
    #[inline]
    pub fn insert_single(&mut self, char_idx: usize, measurable: M) {
        self.try_insert_single(char_idx, measurable).unwrap()
    }

    /// Private internal-only method that does a single insertion of
    /// sufficiently small text.
    ///
    /// This only works correctly for insertion texts smaller than or equal to
    /// `MAX_BYTES - 4`.
    ///
    /// Note that a lot of the complexity in this method comes from avoiding
    /// splitting CRLF pairs and, when possible, avoiding re-scanning text for
    /// text info.  It is otherwise conceptually fairly straightforward.
    fn insert_internal(&mut self, width: usize, mut ins_slice: &[M]) {
        let mut left_seam = false;
        let root_info = self.root.slice_info();

        let (l_info, residual) = Arc::make_mut(&mut self.root).edit_chunk_at_width(
            width,
            root_info,
            |idx, cur_info, leaf_slice| {
                // Find our byte index
                let idx = width_to_idx(leaf_slice, idx);

                // No node splitting
                if (leaf_slice.len() + ins_slice.len()) <= MAX_BYTES {
                    // Calculate new info without doing a full re-scan of cur_text.
                    let new_info = {
                        // Get summed info of current text and to-be-inserted text.
                        #[allow(unused_mut)]
                        let mut info = cur_info + SliceInfo::from_slice(ins_slice);

                        info
                    };
                    // Insert the text and return the new info
                    leaf_slice.insert_slice_split(idx, ins_slice);
                    (new_info, None)
                }
                // We're splitting the node
                else {
                    let r_text = leaf_slice.insert_slice_split(idx, ins_slice);
                    let l_text_info = SliceInfo::from_slice(leaf_slice);
                    if r_text.len() > 0 {
                        let r_text_info = SliceInfo::from_slice(&r_text);
                        (
                            l_text_info,
                            Some((r_text_info, Arc::new(Node::Leaf(r_text)))),
                        )
                    } else {
                        // Leaf couldn't be validly split, so leave it oversized
                        (l_text_info, None)
                    }
                }
            },
        );

        // Handle root splitting, if any.
        if let Some((r_info, r_node)) = residual {
            let mut l_node = Arc::new(Node::new());
            std::mem::swap(&mut l_node, &mut self.root);

            let mut children = Branch::new();
            children.push((l_info, l_node));
            children.push((r_info, r_node));

            *Arc::make_mut(&mut self.root) = Node::Branch(children);
        }
    }

    /// Removes the text in the given char index range.
    ///
    /// Uses range syntax, e.g. `2..7`, `2..`, etc.  The range is in `char`
    /// indices.
    ///
    /// Runs in O(M + log N) time, where N is the length of the `Rope` and M
    /// is the length of the range being removed.
    ///
    /// # Example
    ///
    /// ```
    /// # use ropey::Rope;
    /// let mut rope = Rope::from_str("Hello world!");
    /// rope.remove(5..);
    ///
    /// assert_eq!("Hello", rope);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end, or if the
    /// end is out of bounds (i.e. `end > len_chars()`).
    pub fn remove<R>(&mut self, char_range: R)
    where
        R: RangeBounds<usize>,
    {
        self.try_remove(char_range).unwrap()
    }

    /// Splits the `Rope` at `char_idx`, returning the right part of
    /// the split.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_idx` is out of bounds (i.e. `char_idx > len_chars()`).
    pub fn split_off(&mut self, char_idx: usize) -> Self {
        self.try_split_off(char_idx).unwrap()
    }

    /// Appends a `Rope` to the end of this one, consuming the other `Rope`.
    ///
    /// Runs in O(log N) time.
    pub fn append(&mut self, other: Self) {
        if self.total_width() == 0 {
            // Special case
            let mut other = other;
            std::mem::swap(self, &mut other);
        } else if other.total_width() > 0 {
            let left_info = self.root.slice_info();
            let right_info = other.root.slice_info();

            let l_depth = self.root.depth();
            let r_depth = other.root.depth();

            if l_depth > r_depth {
                let extra =
                    Arc::make_mut(&mut self.root).append_at_depth(other.root, l_depth - r_depth);
                if let Some(node) = extra {
                    let mut children = Branch::new();
                    children.push((self.root.slice_info(), Arc::clone(&self.root)));
                    children.push((node.slice_info(), node));
                    self.root = Arc::new(Node::Branch(children));
                }
            } else {
                let mut other = other;
                let extra = Arc::make_mut(&mut other.root)
                    .prepend_at_depth(Arc::clone(&self.root), r_depth - l_depth);
                if let Some(node) = extra {
                    let mut children = Branch::new();
                    children.push((node.slice_info(), node));
                    children.push((other.root.slice_info(), Arc::clone(&other.root)));
                    other.root = Arc::new(Node::Branch(children));
                }
                *self = other;
            };

            // Fix up any mess left behind.
            let root = Arc::make_mut(&mut self.root);
            if (left_info.len as usize) < MIN_BYTES || (right_info.len as usize) < MIN_BYTES {
                root.fix_tree_seam(left_info.width as usize);
            }
            self.pull_up_singular_nodes();
        }
    }

    //-----------------------------------------------------------------------
    // Index conversion methods

    /// Returns the char index of the given byte.
    ///
    /// Notes:
    ///
    /// - If the byte is in the middle of a multi-byte char, returns the
    ///   index of the char that the byte belongs to.
    /// - `byte_idx` can be one-past-the-end, which will return
    ///   one-past-the-end char index.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_idx` is out of bounds (i.e. `byte_idx > len_bytes()`).
    #[inline]
    pub fn idx_to_width(&self, byte_idx: usize) -> usize {
        self.try_idx_to_width(byte_idx).unwrap()
    }

    /// Returns the byte index of the given char.
    ///
    /// Notes:
    ///
    /// - `char_idx` can be one-past-the-end, which will return
    ///   one-past-the-end byte index.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_idx` is out of bounds (i.e. `char_idx > len_chars()`).
    #[inline]
    pub fn width_to_idx(&self, width: usize) -> usize {
        self.try_width_to_idx(width).unwrap()
    }

    //-----------------------------------------------------------------------
    // Fetch methods

    /// Returns the byte at `byte_idx`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_idx` is out of bounds (i.e. `byte_idx >= len_bytes()`).
    #[inline]
    pub fn from_idx(&self, idx: usize) -> &M {
        // Bounds check
        if let Some(out) = self.get_from_idx(idx) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: byte index {}, Rope byte length {}",
                idx,
                self.total_len()
            );
        }
    }

    /// Returns the char at `char_idx`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_idx` is out of bounds (i.e. `char_idx >= len_chars()`).
    #[inline]
    pub fn from_width(&self, width: usize) -> &M {
        if let Some(out) = self.get_from_width(width) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: char index {}, Rope char length {}",
                width,
                self.total_width()
            );
        }
    }

    /// Returns the chunk containing the given byte index.
    ///
    /// Also returns the byte and char indices of the beginning of the chunk
    /// and the index of the line that the chunk starts on.
    ///
    /// Note: for convenience, a one-past-the-end `byte_idx` returns the last
    /// chunk of the `RopeSlice`.
    ///
    /// The return value is organized as
    /// `(chunk, chunk_byte_idx, chunk_char_idx, chunk_line_idx)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_idx` is out of bounds (i.e. `byte_idx > len_bytes()`).
    #[inline]
    pub fn chunk_at_idx(&self, idx: usize) -> (&[M], usize, usize) {
        // Bounds check
        if let Some(out) = self.get_chunk_at_idx(idx) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: byte index {}, Rope byte length {}",
                idx,
                self.total_len()
            );
        }
    }

    /// Returns the chunk containing the given char index.
    ///
    /// Also returns the byte and char indices of the beginning of the chunk
    /// and the index of the line that the chunk starts on.
    ///
    /// Note: for convenience, a one-past-the-end `char_idx` returns the last
    /// chunk of the `RopeSlice`.
    ///
    /// The return value is organized as
    /// `(chunk, chunk_byte_idx, chunk_char_idx, chunk_line_idx)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_idx` is out of bounds (i.e. `char_idx > len_chars()`).
    #[inline]
    pub fn chunk_at_width(&self, width: usize) -> (&[M], usize, usize) {
        if let Some(out) = self.get_chunk_at_width(width) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: char index {}, Rope char length {}",
                width,
                self.total_width()
            );
        }
    }

    //-----------------------------------------------------------------------
    // Slicing

    /// Gets an immutable slice of the `Rope`, using char indices.
    ///
    /// Uses range syntax, e.g. `2..7`, `2..`, etc.
    ///
    /// # Example
    ///
    /// ```
    /// # use ropey::Rope;
    /// let rope = Rope::from_str("Hello world!");
    /// let slice = rope.slice(..5);
    ///
    /// assert_eq!("Hello", slice);
    /// ```
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end, or if the
    /// end is out of bounds (i.e. `end > len_chars()`).
    #[inline]
    pub fn slice<R>(&self, width_range: R) -> RopeSlice<M>
    where
        R: RangeBounds<usize>,
    {
        self.get_slice(width_range).unwrap()
    }

    /// Gets and immutable slice of the `Rope`, using byte indices.
    ///
    /// Uses range syntax, e.g. `2..7`, `2..`, etc.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The start of the range is greater than the end.
    /// - The end is out of bounds (i.e. `end > len_bytes()`).
    /// - The range doesn't align with char boundaries.
    pub fn idx_slice<R>(&self, idx_range: R) -> RopeSlice<M>
    where
        R: RangeBounds<usize>,
    {
        match self.get_index_slice_impl(idx_range) {
            Ok(s) => return s,
            Err(e) => panic!("byte_slice(): {}", e),
        }
    }

    //-----------------------------------------------------------------------
    // Iterator methods

    /// Creates an iterator over the bytes of the `Rope`.
    ///
    /// Runs in O(log N) time.
    #[inline]
    pub fn iter(&self) -> Iter<M> {
        Iter::new(&self.root)
    }

    /// Creates an iterator over the bytes of the `Rope`, starting at byte
    /// `byte_idx`.
    ///
    /// If `byte_idx == len_bytes()` then an iterator at the end of the
    /// `Rope` is created (i.e. `next()` will return `None`).
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_idx` is out of bounds (i.e. `byte_idx > len_bytes()`).
    #[inline]
    pub fn iter_at_idx(&self, idx: usize) -> Iter<M> {
        if let Some(out) = self.get_iter_at_idx(idx) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: byte index {}, Rope byte length {}",
                idx,
                self.total_len()
            );
        }
    }

    /// Creates an iterator over the chunks of the `Rope`.
    ///
    /// Runs in O(log N) time.
    #[inline]
    pub fn chunks(&self) -> Chunks<M> {
        Chunks::new(&self.root)
    }

    /// Creates an iterator over the chunks of the `Rope`, with the
    /// iterator starting at the chunk containing `byte_idx`.
    ///
    /// Also returns the byte and char indices of the beginning of the first
    /// chunk to be yielded, and the index of the line that chunk starts on.
    ///
    /// If `byte_idx == len_bytes()` an iterator at the end of the `Rope`
    /// (yielding `None` on a call to `next()`) is created.
    ///
    /// The return value is organized as
    /// `(iterator, chunk_byte_idx, chunk_char_idx, chunk_line_idx)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_idx` is out of bounds (i.e. `byte_idx > len_bytes()`).
    #[inline]
    pub fn chunks_at_idx(&self, idx: usize) -> (Chunks<M>, usize, usize) {
        if let Some(out) = self.get_chunks_at_index(idx) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: byte index {}, Rope byte length {}",
                idx,
                self.total_len()
            );
        }
    }

    /// Creates an iterator over the chunks of the `Rope`, with the
    /// iterator starting at the chunk containing `char_idx`.
    ///
    /// Also returns the byte and char indices of the beginning of the first
    /// chunk to be yielded, and the index of the line that chunk starts on.
    ///
    /// If `char_idx == len_chars()` an iterator at the end of the `Rope`
    /// (yielding `None` on a call to `next()`) is created.
    ///
    /// The return value is organized as
    /// `(iterator, chunk_byte_idx, chunk_char_idx, chunk_line_idx)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_idx` is out of bounds (i.e. `char_idx > len_chars()`).
    #[inline]
    pub fn chunks_at_width(&self, width: usize) -> (Chunks<M>, usize, usize) {
        if let Some(out) = self.get_chunks_at_width(width) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: char index {}, Rope char length {}",
                width,
                self.total_width()
            );
        }
    }

    /// Returns true if this rope and `other` point to precisely the same
    /// in-memory data.
    ///
    /// This happens when one of the ropes is a clone of the other and
    /// neither have been modified since then.  Because clones initially
    /// share all the same data, it can be useful to check if they still
    /// point to precisely the same memory as a way of determining
    /// whether they are both still unmodified.
    ///
    /// Note: this is distinct from checking for equality: two ropes can
    /// have the same *contents* (equal) but be stored in different
    /// memory locations (not instances).  Importantly, two clones that
    /// post-cloning are modified identically will *not* be instances
    /// anymore, even though they will have equal contents.
    ///
    /// Runs in O(1) time.
    #[inline]
    pub fn is_instance(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.root, &other.root)
    }

    //-----------------------------------------------------------------------
    // Debugging

    /// NOT PART OF THE PUBLIC API (hidden from docs for a reason!)
    ///
    /// Debugging tool to make sure that all of the meta-data of the
    /// tree is consistent with the actual data.
    #[doc(hidden)]
    pub fn assert_integrity(&self) {
        self.root.assert_integrity();
    }

    /// NOT PART OF THE PUBLIC API (hidden from docs for a reason!)
    ///
    /// Debugging tool to make sure that all of the following invariants
    /// hold true throughout the tree:
    ///
    /// - The tree is the same height everywhere.
    /// - All internal nodes have the minimum number of children.
    /// - All leaf nodes are non-empty.
    /// - CRLF pairs are never split over chunk boundaries.
    #[doc(hidden)]
    pub fn assert_invariants(&self) {
        self.root.assert_balance();
        self.root.assert_node_size(true);
    }

    //-----------------------------------------------------------------------
    // Internal utilities

    /// Iteratively replaces the root node with its child if it only has
    /// one child.
    pub(crate) fn pull_up_singular_nodes(&mut self) {
        while (!self.root.is_leaf()) && self.root.child_count() == 1 {
            let child = if let Node::Branch(ref children) = *self.root {
                Arc::clone(&children.nodes()[0])
            } else {
                unreachable!()
            };

            self.root = child;
        }
    }
}

/// # Non-Panicking
///
/// The methods in this impl block provide non-panicking versions of
/// `Rope`'s panicking methods.  They return either `Option::None` or
/// `Result::Err()` when their panicking counterparts would have panicked.
impl<M> Rope<M>
where
    M: Measurable,
{
    /// Non-panicking version of [`insert()`](Rope::insert).
    #[inline]
    pub fn try_insert(&mut self, char_idx: usize, elements: &[M]) -> Result<()> {
        // Bounds check
        if char_idx <= self.total_width() {
            // We have three cases here:
            // 1. The insertion text is very large, in which case building a new
            //    Rope out of it and splicing it into the existing Rope is most
            //    efficient.
            // 2. The insertion text is somewhat large, in which case splitting it
            //    up into chunks and repeatedly inserting them is the most
            //    efficient.  The splitting is necessary because the insertion code
            //    only works correctly below a certain insertion size.
            // 3. The insertion text is small, in which case we can simply insert
            //    it.
            //
            // Cases #2 and #3 are rolled into one case here, where case #3 just
            // results in the text being "split" into only one chunk.
            //
            // The boundary for what constitutes "very large" text was arrived at
            // experimentally, by testing at what point Rope build + splice becomes
            // faster than split + repeated insert.  This constant is likely worth
            // revisiting from time to time as Ropey evolves.
            if elements.len() > MAX_BYTES * 6 {
                // Case #1: very large text, build rope and splice it in.
                let text_rope = Rope::from_slice(elements);
                let right = self.split_off(char_idx);
                self.append(text_rope);
                self.append(right);
            } else {
                // Cases #2 and #3: split into chunks and repeatedly insert.
                let mut text = elements;
                while !text.is_empty() {
                    // Split a chunk off from the end of the text.
                    // We do this from the end instead of the front so that
                    // the repeated insertions can keep re-using the same
                    // insertion point.
                    let split_idx = text.len() - (MAX_BYTES - 4).min(text.len());
                    let ins_text = &text[split_idx..];
                    text = &text[..split_idx];

                    // Do the insertion.
                    self.insert_internal(char_idx, ins_text);
                }
            }
            Ok(())
        } else {
            Err(Error::CharIndexOutOfBounds(char_idx, self.total_width()))
        }
    }

    /// Non-panicking version of [`insert_char()`](Rope::insert_char).
    #[inline]
    pub fn try_insert_single(&mut self, width: usize, measurable: M) -> Result<()> {
        // Bounds check
        if width <= self.total_width() {
            self.insert_internal(width, &[measurable]);
            Ok(())
        } else {
            Err(Error::CharIndexOutOfBounds(width, self.total_width()))
        }
    }

    /// Non-panicking version of [`remove()`](Rope::remove).
    pub fn try_remove<R>(&mut self, char_range: R) -> Result<()>
    where
        R: RangeBounds<usize>,
    {
        let start_opt = start_bound_to_num(char_range.start_bound());
        let end_opt = end_bound_to_num(char_range.end_bound());
        let start = start_opt.unwrap_or(0);
        let end = end_opt.unwrap_or_else(|| self.total_width());
        if end.max(start) > self.total_width() {
            Err(Error::CharRangeOutOfBounds(
                start_opt,
                end_opt,
                self.total_width(),
            ))
        } else if start > end {
            Err(Error::CharRangeInvalid(start, end))
        } else {
            // A special case that the rest of the logic doesn't handle
            // correctly.
            if start == 0 && end == self.total_width() {
                self.root = Arc::new(Node::new());
                return Ok(());
            }

            let root = Arc::make_mut(&mut self.root);

            let root_info = root.slice_info();
            let (_, needs_fix) = root.remove_range(start, end, root_info);

            if needs_fix {
                root.fix_tree_seam(start);
            }

            self.pull_up_singular_nodes();
            Ok(())
        }
    }

    /// Non-panicking version of [`split_off()`](Rope::split_off).
    pub fn try_split_off(&mut self, char_idx: usize) -> Result<Self> {
        // Bounds check
        if char_idx <= self.total_width() {
            if char_idx == 0 {
                // Special case 1
                let mut new_rope = Rope::new();
                std::mem::swap(self, &mut new_rope);
                Ok(new_rope)
            } else if char_idx == self.total_width() {
                // Special case 2
                Ok(Rope::new())
            } else {
                // Do the split
                let mut new_rope = Rope {
                    root: Arc::new(Arc::make_mut(&mut self.root).split(char_idx)),
                };

                // Fix up the edges
                Arc::make_mut(&mut self.root).zip_fix_right();
                Arc::make_mut(&mut new_rope.root).zip_fix_left();
                self.pull_up_singular_nodes();
                new_rope.pull_up_singular_nodes();

                Ok(new_rope)
            }
        } else {
            Err(Error::CharIndexOutOfBounds(char_idx, self.total_width()))
        }
    }

    /// Non-panicking version of [`byte_to_char()`](Rope::byte_to_char).
    #[inline]
    pub fn try_idx_to_width(&self, idx: usize) -> Result<usize> {
        // Bounds check
        if idx <= self.total_len() {
            let (chunk, b, c) = self.chunk_at_idx(idx);
            Ok(c + idx_to_width(chunk, idx - b))
        } else {
            Err(Error::ByteIndexOutOfBounds(idx, self.total_len()))
        }
    }

    /// Non-panicking version of [`char_to_byte()`](Rope::char_to_byte).
    #[inline]
    pub fn try_width_to_idx(&self, width: usize) -> Result<usize> {
        // Bounds check
        if width <= self.total_width() {
            let (chunk, b, c) = self.chunk_at_width(width);
            Ok(b + width_to_idx(chunk, width - c))
        } else {
            Err(Error::CharIndexOutOfBounds(width, self.total_width()))
        }
    }

    /// Non-panicking version of [`byte()`](Rope::byte).
    #[inline]
    pub fn get_from_idx(&self, idx: usize) -> Option<&M> {
        // Bounds check
        if idx < self.total_len() {
            let (chunk, chunk_byte_idx, _) = self.chunk_at_idx(idx);
            let chunk_rel_byte_idx = idx - chunk_byte_idx;
            Some(&chunk[chunk_rel_byte_idx])
        } else {
            None
        }
    }

    /// Non-panicking version of [`char()`](Rope::char).
    #[inline]
    pub fn get_from_width(&self, width: usize) -> Option<&M> {
        // Bounds check
        if width < self.total_width() {
            let (chunk, _, chunk_width) = self.chunk_at_width(width);
            let byte_idx = width_to_idx(chunk, width - chunk_width);
            Some(&chunk[byte_idx])
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunk_at_byte()`](Rope::chunk_at_byte).
    #[inline]
    pub fn get_chunk_at_idx(&self, byte_idx: usize) -> Option<(&[M], usize, usize)> {
        // Bounds check
        if byte_idx <= self.total_len() {
            let (chunk, info) = self.root.get_chunk_at_idx(byte_idx);
            Some((chunk, info.len as usize, info.width as usize))
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunk_at_char()`](Rope::chunk_at_char).
    #[inline]
    pub fn get_chunk_at_width(&self, width: usize) -> Option<(&[M], usize, usize)> {
        // Bounds check
        if width <= self.total_width() {
            let (chunk, info) = self.root.get_chunk_at_width(width);
            Some((chunk, info.len as usize, info.width as usize))
        } else {
            None
        }
    }

    /// Non-panicking version of [`slice()`](Rope::slice).
    #[inline]
    pub fn get_slice<R>(&self, width_range: R) -> Option<RopeSlice<M>>
    where
        R: RangeBounds<usize>,
    {
        let start = start_bound_to_num(width_range.start_bound()).unwrap_or(0);
        let end = end_bound_to_num(width_range.end_bound()).unwrap_or_else(|| self.total_width());

        // Bounds check
        if start <= end && end <= self.total_width() {
            Some(RopeSlice::new_with_range(&self.root, start, end))
        } else {
            None
        }
    }

    /// Non-panicking version of [`byte_slice()`](Rope::byte_slice).
    #[inline]
    pub fn get_index_slice<R>(&self, byte_range: R) -> Option<RopeSlice<M>>
    where
        R: RangeBounds<usize>,
    {
        self.get_index_slice_impl(byte_range).ok()
    }

    pub(crate) fn get_index_slice_impl<R>(&self, byte_range: R) -> Result<RopeSlice<M>>
    where
        R: RangeBounds<usize>,
    {
        let start_range = start_bound_to_num(byte_range.start_bound());
        let end_range = end_bound_to_num(byte_range.end_bound());

        // Bounds checks.
        match (start_range, end_range) {
            (Some(s), Some(e)) => {
                if s > e {
                    return Err(Error::ByteRangeInvalid(s, e));
                } else if e > self.total_len() {
                    return Err(Error::ByteRangeOutOfBounds(
                        start_range,
                        end_range,
                        self.total_len(),
                    ));
                }
            }
            (Some(s), None) => {
                if s > self.total_len() {
                    return Err(Error::ByteRangeOutOfBounds(
                        start_range,
                        end_range,
                        self.total_len(),
                    ));
                }
            }
            (None, Some(e)) => {
                if e > self.total_len() {
                    return Err(Error::ByteRangeOutOfBounds(
                        start_range,
                        end_range,
                        self.total_len(),
                    ));
                }
            }
            _ => {}
        }

        let (start, end) = (
            start_range.unwrap_or(0),
            end_range.unwrap_or_else(|| self.total_len()),
        );

        RopeSlice::new_with_idx_range(&self.root, start, end).map_err(|e| {
            if let Error::ByteRangeNotCharBoundary(_, _) = e {
                Error::ByteRangeNotCharBoundary(start_range, end_range)
            } else {
                e
            }
        })
    }

    /// Non-panicking version of [`iter_at()`](Self::iter_at).
    #[inline]
    pub fn get_iter_at_idx(&self, idx: usize) -> Option<Iter<M>> {
        // Bounds check
        if idx <= self.total_len() {
            let info = self.root.slice_info();
            Some(Iter::new_with_range(
                &self.root,
                (idx, info.len as usize),
                (0, info.width as usize),
            ))
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunks_at_byte()`](Rope::chunks_at_byte).
    #[inline]
    pub fn get_chunks_at_index(&self, idx: usize) -> Option<(Chunks<M>, usize, usize)> {
        // Bounds check
        if idx <= self.total_len() {
            Some(Chunks::new_with_range_at_idx(
                &self.root,
                idx,
                (0, self.total_len()),
                (0, self.total_width()),
            ))
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunks_at_char()`](Rope::chunks_at_char).
    #[inline]
    pub fn get_chunks_at_width(&self, char_idx: usize) -> Option<(Chunks<M>, usize, usize)> {
        // Bounds check
        if char_idx <= self.total_width() {
            Some(Chunks::new_with_range_at_width(
                &self.root,
                char_idx,
                (0, self.total_len()),
                (0, self.total_width()),
            ))
        } else {
            None
        }
    }
}

//==============================================================
// Conversion impls

impl<'a, M> From<&'a [M]> for Rope<M>
where
    M: Measurable,
{
    #[inline]
    fn from(slice: &'a [M]) -> Self {
        Rope::from_slice(slice)
    }
}

impl<'a, M> From<std::borrow::Cow<'a, [M]>> for Rope<M>
where
    M: Measurable,
{
    #[inline]
    fn from(slice: std::borrow::Cow<'a, [M]>) -> Self {
        Rope::from_slice(&slice)
    }
}

impl<M> From<Vec<M>> for Rope<M>
where
    M: Measurable,
{
    #[inline]
    fn from(slice: Vec<M>) -> Self {
        Rope::from_slice(&slice)
    }
}

/// Will share data where possible.
///
/// Runs in O(log N) time.
impl<'a, M> From<RopeSlice<'a, M>> for Rope<M>
where
    M: Measurable,
{
    fn from(s: RopeSlice<'a, M>) -> Self {
        use crate::slice::RSEnum;
        match s {
            RopeSlice(RSEnum::Full {
                node,
                start_info,
                end_info,
            }) => {
                let mut rope = Rope {
                    root: Arc::clone(node),
                };

                // Chop off right end if needed
                if end_info.width < node.slice_info().width {
                    {
                        let root = Arc::make_mut(&mut rope.root);
                        root.split(end_info.width as usize);
                        root.zip_fix_right();
                    }
                    rope.pull_up_singular_nodes();
                }

                // Chop off left end if needed
                if start_info.width > 0 {
                    {
                        let root = Arc::make_mut(&mut rope.root);
                        *root = root.split(start_info.width as usize);
                        root.zip_fix_left();
                    }
                    rope.pull_up_singular_nodes();
                }

                // Return the rope
                rope
            }
            RopeSlice(RSEnum::Light { slice: text, .. }) => Rope::from_slice(text),
        }
    }
}

impl<M> From<Rope<M>> for Vec<M>
where
    M: Measurable,
{
    #[inline]
    fn from(r: Rope<M>) -> Self {
        Vec::from(&r)
    }
}

impl<'a, M> From<&'a Rope<M>> for Vec<M>
where
    M: Measurable,
{
    #[inline]
    fn from(r: &'a Rope<M>) -> Self {
        let mut vec = Vec::with_capacity(r.total_len());
        vec.extend(
            r.chunks()
                .map(|chunk| chunk.iter())
                .flatten()
                .map(|measurable| *measurable),
        );
        vec
    }
}

impl<'a, M> From<Rope<M>> for std::borrow::Cow<'a, [M]>
where
    M: Measurable,
{
    #[inline]
    fn from(r: Rope<M>) -> Self {
        std::borrow::Cow::Owned(Vec::from(r))
    }
}

/// Attempts to borrow the contents of the `Rope`, but will convert to an
/// owned string if the contents is not contiguous in memory.
///
/// Runs in best case O(1), worst case O(N).
impl<'a, M> From<&'a Rope<M>> for std::borrow::Cow<'a, [M]>
where
    M: Measurable,
{
    #[inline]
    fn from(r: &'a Rope<M>) -> Self {
        if let Node::Leaf(ref text) = *r.root {
            std::borrow::Cow::Borrowed(text)
        } else {
            std::borrow::Cow::Owned(Vec::from(r))
        }
    }
}

impl<'a, M> FromIterator<&'a [M]> for Rope<M>
where
    M: Measurable,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a [M]>,
    {
        let mut builder = RopeBuilder::new();
        for chunk in iter {
            builder.append(chunk);
        }
        builder.finish()
    }
}

impl<'a, M> FromIterator<std::borrow::Cow<'a, [M]>> for Rope<M>
where
    M: Measurable,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = std::borrow::Cow<'a, [M]>>,
    {
        let mut builder = RopeBuilder::new();
        for chunk in iter {
            builder.append(&chunk);
        }
        builder.finish()
    }
}

impl<'a, M> FromIterator<Vec<M>> for Rope<M>
where
    M: Measurable,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Vec<M>>,
    {
        let mut builder = RopeBuilder::new();
        for chunk in iter {
            builder.append(&chunk);
        }
        builder.finish()
    }
}

//==============================================================
// Other impls

impl<M> std::fmt::Debug for Rope<M>
where
    M: Measurable + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_list().entries(self.chunks()).finish()
    }
}

impl<M> std::fmt::Display for Rope<M>
where
    M: Measurable + Debug,
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_list().entries(self.chunks()).finish()
    }
}

impl<M> std::default::Default for Rope<M>
where
    M: Measurable,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<M> std::cmp::Eq for Rope<M> where M: Measurable + Eq {}

impl<M> std::cmp::PartialEq<Rope<M>> for Rope<M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        self.slice(..) == other.slice(..)
    }
}

impl<'a, M> std::cmp::PartialEq<&'a [M]> for Rope<M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &&'a [M]) -> bool {
        self.slice(..) == *other
    }
}

impl<'a, M> std::cmp::PartialEq<Rope<M>> for &'a [M]
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        *self == other.slice(..)
    }
}

impl<M> std::cmp::PartialEq<[M]> for Rope<M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &[M]) -> bool {
        self.slice(..) == other
    }
}

impl<M> std::cmp::PartialEq<Rope<M>> for [M]
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        self == other.slice(..)
    }
}

impl<'a, M> std::cmp::PartialEq<Vec<M>> for Rope<M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Vec<M>) -> bool {
        self.slice(..) == other.as_slice()
    }
}

impl<'a, M> std::cmp::PartialEq<Rope<M>> for Vec<M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        self.as_slice() == other.slice(..)
    }
}

impl<'a, M> std::cmp::PartialEq<std::borrow::Cow<'a, [M]>> for Rope<M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &std::borrow::Cow<'a, [M]>) -> bool {
        self.slice(..) == **other
    }
}

impl<'a, M> std::cmp::PartialEq<Rope<M>> for std::borrow::Cow<'a, [M]>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        **self == other.slice(..)
    }
}

impl<M> std::cmp::Ord for Rope<M>
where
    M: Measurable + Ord,
{
    #[inline]
    fn cmp(&self, other: &Rope<M>) -> std::cmp::Ordering {
        self.slice(..).cmp(&other.slice(..))
    }
}

impl<M> std::cmp::PartialOrd<Rope<M>> for Rope<M>
where
    M: Measurable + PartialOrd + Ord,
{
    #[inline]
    fn partial_cmp(&self, other: &Rope<M>) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

//==============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slice_utils::*;
    use std::hash::{Hash, Hasher};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Test {
        Foo,
        Bar,
        Baz(usize),
        Qux,
        Quux,
    }

    impl Measurable for Test {
        fn width(&self) -> usize {
            match self {
                Test::Foo => 1,
                Test::Bar => 2,
                Test::Baz(width) => *width,
                Test::Qux => 0,
                Test::Quux => 0,
            }
        }
    }

    use self::Test::*;
    // 7 elements, total width of 10.
    const NATURAL_WIDTH: &[Test] = &[Foo, Bar, Foo, Bar, Foo, Bar, Foo];
    // 7 elements, total width of 40.
    const VAR_WIDTH: &[Test] = &[Foo, Bar, Baz(4), Bar, Baz(30), Foo];
    // 7 elements, total width of 5.
    const ZERO_WIDTH: &[Test] = &[Qux, Quux, Bar, Qux, Quux, Baz(3), Qux];

    #[test]
    fn new_01() {
        let r = Rope::<Test>::new();
        assert_eq!(r, [].as_slice());

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn from_slice() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        assert_eq!(r, NATURAL_WIDTH);

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn len_bytes_01() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        assert_eq!(r.total_len(), 127);
    }

    #[test]
    fn len_bytes_02() {
        let r = Rope::<Test>::from_slice(&[]);
        assert_eq!(r.total_len(), 0);
    }

    #[test]
    fn len_chars_01() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        assert_eq!(r.total_width(), 103);
    }

    #[test]
    fn len_chars_02() {
        let r = Rope::<Test>::from_slice(&[]);
        assert_eq!(r.total_width(), 0);
    }

    #[test]
    fn insert_01() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);
        r.insert(3, &[Foo, Bar, Baz(3)]);

        assert_eq!(
            r,
            "HelAAlo there!  How're you doing?  It's \
             a fine day, isn't it?  Aren't you glad \
             we're alive?  こんにちは、みんなさん！"
        );

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn insert_02() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);
        r.insert(0, &[Foo, Bar, Baz(3)]);

        assert_eq!(
            r,
            "AAHello there!  How're you doing?  It's \
             a fine day, isn't it?  Aren't you glad \
             we're alive?  こんにちは、みんなさん！"
        );

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn insert_03() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);
        r.insert(103, &[Foo, Bar, Baz(3)]);

        assert_eq!(
            r,
            "Hello there!  How're you doing?  It's \
             a fine day, isn't it?  Aren't you glad \
             we're alive?  こんにちは、みんなさん！AA"
        );

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn insert_04() {
        let mut r = Rope::new();
        r.insert(0, "He");
        r.insert(2, "l");
        r.insert(3, "l");
        r.insert(4, "o w");
        r.insert(7, "o");
        r.insert(8, "rl");
        r.insert(10, "d!");
        r.insert(3, "zopter");

        assert_eq!("Helzopterlo world!", r);

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn insert_05() {
        let mut r = Rope::new();
        r.insert(0, "こんいちは、みんなさん！");
        r.insert(7, "zopter");
        assert_eq!("こんいちは、みzopterんなさん！", r);

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn insert_06() {
        let mut r = Rope::new();
        r.insert(0, "こ");
        r.insert(1, "ん");
        r.insert(2, "い");
        r.insert(3, "ち");
        r.insert(4, "は");
        r.insert(5, "、");
        r.insert(6, "み");
        r.insert(7, "ん");
        r.insert(8, "な");
        r.insert(9, "さ");
        r.insert(10, "ん");
        r.insert(11, "！");
        r.insert(7, "zopter");
        assert_eq!("こんいちは、みzopterんなさん！", r);

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn insert_char_01() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);
        r.insert_char(3, 'A');
        r.insert_char(12, '!');
        r.insert_char(12, '!');

        assert_eq!(
            r,
            "HelAlo there!!!  How're you doing?  It's \
             a fine day, isn't it?  Aren't you glad \
             we're alive?  こんにちは、みんなさん！"
        );

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn insert_char_02() {
        let mut r = Rope::new();

        r.insert_single(0, '！');
        r.insert_single(0, 'こ');
        r.insert_single(1, 'ん');
        r.insert_single(2, 'い');
        r.insert_single(3, 'ち');
        r.insert_single(4, 'は');
        r.insert_single(5, '、');
        r.insert_single(6, 'み');
        r.insert_single(7, 'ん');
        r.insert_single(8, 'な');
        r.insert_single(9, 'さ');
        r.insert_single(10, 'ん');
        assert_eq!("こんいちは、みんなさん！", r);

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn remove_01() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);

        r.remove(5..11);
        r.remove(24..31);
        r.remove(19..25);
        r.remove(75..79);
        assert_eq!(
            r,
            "Hello!  How're you \
             a fine day, isn't it?  Aren't you glad \
             we're alive?  こんにんなさん！"
        );

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn remove_02() {
        let mut r = Rope::from_slice("\r\n\r\n\r\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n");

        // Make sure CRLF pairs get merged properly, via
        // assert_invariants() below.
        r.remove(3..6);
        assert_eq!(r, "\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n");

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn remove_03() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);

        // Make sure removing nothing actually does nothing.
        r.remove(45..45);
        assert_eq!(r, NATURAL_WIDTH);

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn remove_04() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);

        // Make sure removing everything works.
        r.remove(0..103);
        assert_eq!(r, "");

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn remove_05() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);

        // Make sure removing a large range works.
        r.remove(3..100);
        assert_eq!(r, "Helさん！");

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn remove_06() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);
        #[allow(clippy::reversed_empty_ranges)]
        r.remove(56..55); // Wrong ordering of start/end on purpose.
    }

    #[test]
    #[should_panic]
    fn remove_07() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);
        r.remove(102..104); // Removing past the end
    }

    #[test]
    #[should_panic]
    fn remove_08() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);
        r.remove(103..104); // Removing past the end
    }

    #[test]
    #[should_panic]
    fn remove_09() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);
        r.remove(104..104); // Removing past the end
    }

    #[test]
    #[should_panic]
    fn remove_10() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);
        r.remove(104..105); // Removing past the end
    }

    #[test]
    fn split_off_01() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);

        let r2 = r.split_off(50);
        assert_eq!(
            r,
            "Hello there!  How're you doing?  It's \
             a fine day, "
        );
        assert_eq!(
            r2,
            "isn't it?  Aren't you glad \
             we're alive?  こんにちは、みんなさん！"
        );

        r.assert_integrity();
        r2.assert_integrity();
        r.assert_invariants();
        r2.assert_invariants();
    }

    #[test]
    fn split_off_02() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);

        let r2 = r.split_off(1);
        assert_eq!(r, "H");
        assert_eq!(
            r2,
            "ello there!  How're you doing?  It's \
             a fine day, isn't it?  Aren't you glad \
             we're alive?  こんにちは、みんなさん！"
        );

        r.assert_integrity();
        r2.assert_integrity();
        r.assert_invariants();
        r2.assert_invariants();
    }

    #[test]
    fn split_off_03() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);

        let r2 = r.split_off(102);
        assert_eq!(
            r,
            "Hello there!  How're you doing?  It's \
             a fine day, isn't it?  Aren't you glad \
             we're alive?  こんにちは、みんなさん"
        );
        assert_eq!(r2, "！");

        r.assert_integrity();
        r2.assert_integrity();
        r.assert_invariants();
        r2.assert_invariants();
    }

    #[test]
    fn split_off_04() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);

        let r2 = r.split_off(0);
        assert_eq!(r, "");
        assert_eq!(
            r2,
            "Hello there!  How're you doing?  It's \
             a fine day, isn't it?  Aren't you glad \
             we're alive?  こんにちは、みんなさん！"
        );

        r.assert_integrity();
        r2.assert_integrity();
        r.assert_invariants();
        r2.assert_invariants();
    }

    #[test]
    fn split_off_05() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);

        let r2 = r.split_off(103);
        assert_eq!(
            r,
            "Hello there!  How're you doing?  It's \
             a fine day, isn't it?  Aren't you glad \
             we're alive?  こんにちは、みんなさん！"
        );
        assert_eq!(r2, "");

        r.assert_integrity();
        r2.assert_integrity();
        r.assert_invariants();
        r2.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn split_off_06() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);
        r.split_off(104); // One past the end of the rope
    }

    #[test]
    fn append_01() {
        let mut r = Rope::from_slice(
            "Hello there!  How're you doing?  It's \
             a fine day, isn't ",
        );
        let r2 = Rope::from_slice(
            "it?  Aren't you glad \
             we're alive?  こんにちは、みんなさん！",
        );

        r.append(r2);
        assert_eq!(r, NATURAL_WIDTH);

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn append_02() {
        let mut r = Rope::from_slice(
            "Hello there!  How're you doing?  It's \
             a fine day, isn't it?  Aren't you glad \
             we're alive?  こんに",
        );
        let r2 = Rope::from_slice("ちは、みんなさん！");

        r.append(r2);
        assert_eq!(r, NATURAL_WIDTH);

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn append_03() {
        let mut r = Rope::from_slice(
            "Hello there!  How're you doing?  It's \
             a fine day, isn't it?  Aren't you glad \
             we're alive?  こんにちは、みんなさん",
        );
        let r2 = Rope::from_slice("！");

        r.append(r2);
        assert_eq!(r, NATURAL_WIDTH);

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn append_04() {
        let mut r = Rope::from_slice("H");
        let r2 = Rope::from_slice(
            "ello there!  How're you doing?  It's \
             a fine day, isn't it?  Aren't you glad \
             we're alive?  こんにちは、みんなさん！",
        );

        r.append(r2);
        assert_eq!(r, NATURAL_WIDTH);

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn append_05() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);
        let r2 = Rope::from_slice("");

        r.append(r2);
        assert_eq!(r, NATURAL_WIDTH);

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    fn append_06() {
        let mut r = Rope::from_slice("");
        let r2 = Rope::from_slice(NATURAL_WIDTH);

        r.append(r2);
        assert_eq!(r, NATURAL_WIDTH);

        r.assert_integrity();
        r.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn byte_to_line_04() {
        let r = Rope::from_slice(VAR_WIDTH);
        r.byte_to_line(125);
    }

    #[test]
    fn char_to_byte_01() {
        let r = Rope::from_slice(NATURAL_WIDTH);

        assert_eq!(0, r.char_to_byte(0));
        assert_eq!(1, r.char_to_byte(1));
        assert_eq!(2, r.char_to_byte(2));

        assert_eq!(91, r.char_to_byte(91));
        assert_eq!(94, r.char_to_byte(92));
        assert_eq!(97, r.char_to_byte(93));
        assert_eq!(100, r.char_to_byte(94));

        assert_eq!(124, r.char_to_byte(102));
        assert_eq!(127, r.char_to_byte(103));
    }

    #[test]
    fn char_to_line_01() {
        let r = Rope::from_slice(VAR_WIDTH);

        assert_eq!(0, r.char_to_line(0));
        assert_eq!(0, r.char_to_line(1));

        assert_eq!(0, r.char_to_line(31));
        assert_eq!(1, r.char_to_line(32));
        assert_eq!(1, r.char_to_line(33));

        assert_eq!(1, r.char_to_line(58));
        assert_eq!(2, r.char_to_line(59));
        assert_eq!(2, r.char_to_line(60));

        assert_eq!(2, r.char_to_line(87));
        assert_eq!(3, r.char_to_line(88));
        assert_eq!(3, r.char_to_line(89));
        assert_eq!(3, r.char_to_line(100));
    }

    #[test]
    fn char_to_line_02() {
        let r = Rope::from_slice("");
        assert_eq!(0, r.char_to_line(0));
    }

    #[test]
    fn char_to_line_03() {
        let r = Rope::from_slice("Hi there\n");
        assert_eq!(0, r.char_to_line(0));
        assert_eq!(0, r.char_to_line(8));
        assert_eq!(1, r.char_to_line(9));
    }

    #[test]
    #[should_panic]
    fn char_to_line_04() {
        let r = Rope::from_slice(VAR_WIDTH);
        r.char_to_line(101);
    }

    #[test]
    fn line_to_byte_01() {
        let r = Rope::from_slice(VAR_WIDTH);

        assert_eq!(0, r.line_to_byte(0));
        assert_eq!(32, r.line_to_byte(1));
        assert_eq!(59, r.line_to_byte(2));
        assert_eq!(88, r.line_to_byte(3));
        assert_eq!(124, r.line_to_byte(4));
    }

    #[test]
    fn line_to_byte_02() {
        let r = Rope::from_slice("");
        assert_eq!(0, r.line_to_byte(0));
        assert_eq!(0, r.line_to_byte(1));
    }

    #[test]
    #[should_panic]
    fn line_to_byte_03() {
        let r = Rope::from_slice(VAR_WIDTH);
        r.line_to_byte(5);
    }

    #[test]
    fn line_to_char_01() {
        let r = Rope::from_slice(VAR_WIDTH);

        assert_eq!(0, r.line_to_char(0));
        assert_eq!(32, r.line_to_char(1));
        assert_eq!(59, r.line_to_char(2));
        assert_eq!(88, r.line_to_char(3));
        assert_eq!(100, r.line_to_char(4));
    }

    #[test]
    fn line_to_char_02() {
        let r = Rope::from_slice("");
        assert_eq!(0, r.line_to_char(0));
        assert_eq!(0, r.line_to_char(1));
    }

    #[test]
    #[should_panic]
    fn line_to_char_03() {
        let r = Rope::from_slice(VAR_WIDTH);
        r.line_to_char(5);
    }

    #[test]
    fn char_to_utf16_cu_01() {
        let r = Rope::from_slice("");
        assert_eq!(0, r.char_to_utf16_cu(0));
    }

    #[test]
    #[should_panic]
    fn char_to_utf16_cu_02() {
        let r = Rope::from_slice("");
        r.char_to_utf16_cu(1);
    }

    #[test]
    fn char_to_utf16_cu_03() {
        let r = Rope::from_slice(ZERO_WIDTH);

        assert_eq!(0, r.char_to_utf16_cu(0));

        assert_eq!(12, r.char_to_utf16_cu(12));
        assert_eq!(14, r.char_to_utf16_cu(13));

        assert_eq!(33, r.char_to_utf16_cu(32));
        assert_eq!(35, r.char_to_utf16_cu(33));

        assert_eq!(63, r.char_to_utf16_cu(61));
        assert_eq!(65, r.char_to_utf16_cu(62));

        assert_eq!(95, r.char_to_utf16_cu(92));
        assert_eq!(97, r.char_to_utf16_cu(93));

        assert_eq!(111, r.char_to_utf16_cu(107));
    }

    #[test]
    #[should_panic]
    fn char_to_utf16_cu_04() {
        let r = Rope::from_slice(ZERO_WIDTH);
        r.char_to_utf16_cu(108);
    }

    #[test]
    fn utf16_cu_to_char_01() {
        let r = Rope::from_slice("");
        assert_eq!(0, r.utf16_cu_to_char(0));
    }

    #[test]
    #[should_panic]
    fn utf16_cu_to_char_02() {
        let r = Rope::from_slice("");
        r.utf16_cu_to_char(1);
    }

    #[test]
    fn utf16_cu_to_char_03() {
        let r = Rope::from_slice(ZERO_WIDTH);

        assert_eq!(0, r.utf16_cu_to_char(0));

        assert_eq!(12, r.utf16_cu_to_char(12));
        assert_eq!(12, r.utf16_cu_to_char(13));
        assert_eq!(13, r.utf16_cu_to_char(14));

        assert_eq!(32, r.utf16_cu_to_char(33));
        assert_eq!(32, r.utf16_cu_to_char(34));
        assert_eq!(33, r.utf16_cu_to_char(35));

        assert_eq!(61, r.utf16_cu_to_char(63));
        assert_eq!(61, r.utf16_cu_to_char(64));
        assert_eq!(62, r.utf16_cu_to_char(65));

        assert_eq!(92, r.utf16_cu_to_char(95));
        assert_eq!(92, r.utf16_cu_to_char(96));
        assert_eq!(93, r.utf16_cu_to_char(97));

        assert_eq!(107, r.utf16_cu_to_char(111));
    }

    #[test]
    #[should_panic]
    fn utf16_cu_to_char_04() {
        let r = Rope::from_slice(ZERO_WIDTH);
        r.utf16_cu_to_char(112);
    }

    #[test]
    fn byte_01() {
        let r = Rope::from_slice(NATURAL_WIDTH);

        assert_eq!(r.byte(0), b'H');

        // UTF-8 for "wide exclamation mark"
        assert_eq!(r.byte(124), 0xEF);
        assert_eq!(r.byte(125), 0xBC);
        assert_eq!(r.byte(126), 0x81);
    }

    #[test]
    #[should_panic]
    fn byte_02() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        r.byte(127);
    }

    #[test]
    #[should_panic]
    fn byte_03() {
        let r = Rope::from_slice("");
        r.byte(0);
    }

    #[test]
    fn char_01() {
        let r = Rope::from_slice(NATURAL_WIDTH);

        assert_eq!(r.char(0), 'H');
        assert_eq!(r.char(10), 'e');
        assert_eq!(r.char(18), 'r');
        assert_eq!(r.char(102), '！');
    }

    #[test]
    #[should_panic]
    fn char_02() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        r.char(103);
    }

    #[test]
    #[should_panic]
    fn char_03() {
        let r = Rope::from_slice("");
        r.char(0);
    }

    #[test]
    fn chunk_at_index() {
        let r = Rope::from_slice(VAR_WIDTH);
        let mut t = VAR_WIDTH;

        let mut last_chunk = "";
        for i in 0..r.total_len() {
            let (chunk, b, c, l) = r.chunk_at_byte(i);
            assert_eq!(c, idx_to_width(VAR_WIDTH, b));
            if chunk != last_chunk {
                assert_eq!(chunk, &t[..chunk.len()]);
                t = &t[chunk.len()..];
                last_chunk = chunk;
            }

            let c1 = {
                let i2 = idx_to_width(VAR_WIDTH, i);
                VAR_WIDTH.chars().nth(i2).unwrap()
            };
            let c2 = {
                let i2 = i - b;
                let i3 = idx_to_width(chunk, i2);
                chunk.chars().nth(i3).unwrap()
            };
            assert_eq!(c1, c2);
        }
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn chunk_at_char() {
        let r = Rope::from_slice(VAR_WIDTH);
        let mut t = VAR_WIDTH;

        let mut last_chunk = "";
        for i in 0..r.total_width() {
            let (chunk, b, c, l) = r.chunk_at_width(i);
            assert_eq!(b, idx_to_width(VAR_WIDTH, c));
            if chunk != last_chunk {
                assert_eq!(chunk, &t[..chunk.len()]);
                t = &t[chunk.len()..];
                last_chunk = chunk;
            }

            let c1 = VAR_WIDTH.chars().nth(i).unwrap();
            let c2 = {
                let i2 = i - c;
                chunk.chars().nth(i2).unwrap()
            };
            assert_eq!(c1, c2);
        }
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn slice_01() {
        let r = Rope::from_slice(NATURAL_WIDTH);

        let s = r.slice(0..r.total_width());

        assert_eq!(NATURAL_WIDTH, s);
    }

    #[test]
    fn slice_02() {
        let r = Rope::from_slice(NATURAL_WIDTH);

        let s = r.slice(5..21);

        assert_eq!(&NATURAL_WIDTH[5..21], s);
    }

    #[test]
    fn slice_03() {
        let r = Rope::from_slice(NATURAL_WIDTH);

        let s = r.slice(31..97);

        assert_eq!(&NATURAL_WIDTH[31..109], s);
    }

    #[test]
    fn slice_04() {
        let r = Rope::from_slice(NATURAL_WIDTH);

        let s = r.slice(53..53);

        assert_eq!("", s);
    }

    #[test]
    #[should_panic]
    fn slice_05() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        #[allow(clippy::reversed_empty_ranges)]
        r.slice(53..52); // Wrong ordering on purpose.
    }

    #[test]
    #[should_panic]
    fn slice_06() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        r.slice(102..104);
    }

    #[test]
    fn byte_slice_01() {
        let r = Rope::from_slice(NATURAL_WIDTH);

        let s = r.byte_slice(0..r.total_len());

        assert_eq!(NATURAL_WIDTH, s);
    }

    #[test]
    fn byte_slice_02() {
        let r = Rope::from_slice(NATURAL_WIDTH);

        let s = r.byte_slice(5..21);

        assert_eq!(&NATURAL_WIDTH[5..21], s);
    }

    #[test]
    fn byte_slice_03() {
        let r = Rope::from_slice(NATURAL_WIDTH);

        let s = r.byte_slice(31..97);

        assert_eq!(&NATURAL_WIDTH[31..97], s);
    }

    #[test]
    fn byte_slice_04() {
        let r = Rope::from_slice(NATURAL_WIDTH);

        let s = r.byte_slice(53..53);

        assert_eq!("", s);
    }

    #[test]
    #[should_panic]
    fn byte_slice_05() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        #[allow(clippy::reversed_empty_ranges)]
        r.byte_slice(53..52); // Wrong ordering on purpose.
    }

    #[test]
    #[should_panic]
    fn byte_slice_06() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        r.byte_slice(20..128);
    }

    #[test]
    #[should_panic]
    fn byte_slice_07() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        // Not a char boundary.
        r.byte_slice(..96);
    }

    #[test]
    #[should_panic]
    fn byte_slice_08() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        // Not a char boundary.
        r.byte_slice(96..);
    }

    #[test]
    fn eq_rope_01() {
        let r = Rope::from_slice("");

        assert_eq!(r, r);
    }

    #[test]
    fn eq_rope_02() {
        let r = Rope::from_slice(NATURAL_WIDTH);

        assert_eq!(r, r);
    }

    #[test]
    fn eq_rope_03() {
        let r1 = Rope::from_slice(NATURAL_WIDTH);
        let mut r2 = r1.clone();
        r2.remove(26..27);
        r2.insert(26, "z");

        assert_ne!(r1, r2);
    }

    #[test]
    fn eq_rope_04() {
        let r = Rope::from_slice("");

        assert_eq!(r, "");
        assert_eq!("", r);
    }

    #[test]
    fn eq_rope_05() {
        let r = Rope::from_slice(NATURAL_WIDTH);

        assert_eq!(r, NATURAL_WIDTH);
        assert_eq!(NATURAL_WIDTH, r);
    }

    #[test]
    fn eq_rope_06() {
        let mut r = Rope::from_slice(NATURAL_WIDTH);
        r.remove(26..27);
        r.insert(26, "z");

        assert_ne!(r, NATURAL_WIDTH);
        assert_ne!(NATURAL_WIDTH, r);
    }

    #[test]
    fn eq_rope_07() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        let s: String = NATURAL_WIDTH.into();

        assert_eq!(r, s);
        assert_eq!(s, r);
    }

    #[test]
    fn to_string_01() {
        let r = Rope::from_slice(NATURAL_WIDTH);
        let s: String = (&r).into();

        assert_eq!(r, s);
    }

    #[test]
    fn to_cow_01() {
        use std::borrow::Cow;
        let r = Rope::from_slice(NATURAL_WIDTH);
        let cow: Cow<str> = (&r).into();

        assert_eq!(r, cow);
    }

    #[test]
    fn to_cow_02() {
        use std::borrow::Cow;
        let r = Rope::from_slice(NATURAL_WIDTH);
        let cow: Cow<str> = (r.clone()).into();

        assert_eq!(r, cow);
    }

    #[test]
    fn to_cow_03() {
        use std::borrow::Cow;
        let r = Rope::from_slice("a");
        let cow: Cow<str> = (&r).into();

        // Make sure it's borrowed.
        if let Cow::Owned(_) = cow {
            panic!("Small Cow conversions should result in a borrow.");
        }

        assert_eq!(r, cow);
    }

    #[test]
    fn from_rope_slice_01() {
        let r1 = Rope::from_slice(NATURAL_WIDTH);
        let s = r1.slice(..);
        let r2: Rope = s.into();

        assert_eq!(r1, r2);
        assert_eq!(s, r2);
    }

    #[test]
    fn from_rope_slice_02() {
        let r1 = Rope::from_slice(NATURAL_WIDTH);
        let s = r1.slice(0..24);
        let r2: Rope = s.into();

        assert_eq!(s, r2);
    }

    #[test]
    fn from_rope_slice_03() {
        let r1 = Rope::from_slice(NATURAL_WIDTH);
        let s = r1.slice(13..89);
        let r2: Rope = s.into();

        assert_eq!(s, r2);
    }

    #[test]
    fn from_rope_slice_04() {
        let r1 = Rope::from_slice(NATURAL_WIDTH);
        let s = r1.slice(13..41);
        let r2: Rope = s.into();

        assert_eq!(s, r2);
    }

    #[test]
    fn from_iter_01() {
        let r1 = Rope::from_slice(NATURAL_WIDTH);
        let r2: Rope = Rope::from_iter(r1.chunks());

        assert_eq!(r1, r2);
    }

    #[test]
    fn hash_01() {
        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        let r1 = Rope::from_slice("Hello there!");
        let mut r2 = Rope::from_slice("Hlotee");
        r2.insert_char(3, ' ');
        r2.insert_char(7, '!');
        r2.insert_char(1, 'e');
        r2.insert_char(3, 'l');
        r2.insert_char(7, 'h');
        r2.insert_char(9, 'r');

        r1.hash(&mut h1);
        r2.hash(&mut h2);

        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn hash_02() {
        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        let r1 = Rope::from_slice("Hello there!");
        let r2 = Rope::from_slice("Hello there.");

        r1.hash(&mut h1);
        r2.hash(&mut h2);

        assert_ne!(h1.finish(), h2.finish());
    }

    #[test]
    fn hash_03() {
        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        let r = Rope::from_slice("Hello there!");
        let s = [Rope::from_slice("Hello "), Rope::from_slice("there!")];

        r.hash(&mut h1);
        Rope::hash_slice(&s, &mut h2);

        assert_ne!(h1.finish(), h2.finish());
    }

    #[test]
    fn is_instance_01() {
        let r = Rope::from_slice("Hello there!");
        let mut c1 = r.clone();
        let c2 = c1.clone();

        assert!(r.is_instance(&c1));
        assert!(r.is_instance(&c2));
        assert!(c1.is_instance(&c2));

        c1.insert(0, "Oh! ");

        assert!(!r.is_instance(&c1));
        assert!(r.is_instance(&c2));
        assert!(!c1.is_instance(&c2));
    }

    // Iterator tests are in the iter module
}
