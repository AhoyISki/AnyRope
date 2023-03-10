use std::fmt::Debug;
use std::iter::FromIterator;
use std::ops::RangeBounds;
use std::sync::Arc;

use crate::iter::{Chunks, Iter};
use crate::rope_builder::RopeBuilder;
use crate::slice::RopeSlice;
use crate::slice_utils::{index_to_width, width_to_index};
use crate::tree::{Branch, Node, SliceInfo, MAX_BYTES, MIN_BYTES};
use crate::{end_bound_to_num, start_bound_to_num, Error, Result};

pub trait Measurable: Clone + Copy + Debug {
    /// The width of this element, it need not be the actual lenght in bytes, but just a
    /// representative value, to be fed to the [Rope][crate::rope::Rope].
    fn width(&self) -> usize;
}

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
    pub fn len(&self) -> usize {
        self.root.len()
    }

    /// Total number of chars in the `Rope`.
    ///
    /// Runs in O(1) time.
    #[inline]
    pub fn width(&self) -> usize {
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

    /// Inserts `text` at char index `char_index`.
    ///
    /// Runs in O(M + log N) time, where N is the length of the `Rope` and M
    /// is the length of `text`.
    ///
    /// # Panics
    ///
    /// Panics if `char_index` is out of bounds (i.e. `char_index > len_chars()`).
    #[inline]
    pub fn insert_slice(&mut self, char_index: usize, slice: &[M]) {
        // Bounds check
        self.try_insert(char_index, slice).unwrap()
    }

    /// Inserts a single char `ch` at char index `char_index`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_index` is out of bounds (i.e. `char_index > len_chars()`).
    #[inline]
    pub fn insert(&mut self, char_index: usize, measurable: M) {
        self.try_insert_single(char_index, measurable).unwrap()
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
    fn insert_internal(&mut self, width: usize, ins_slice: &[M]) {
        let root_info = self.root.slice_info();

        let (l_info, residual) = Arc::make_mut(&mut self.root).edit_chunk_at_width(
            width,
            root_info,
            |index, cur_info, leaf_slice| {
                // Find our byte index
                let index = width_to_index(leaf_slice, index);

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
                    leaf_slice.insert_slice_split(index, ins_slice);
                    (new_info, None)
                }
                // We're splitting the node
                else {
                    let r_text = leaf_slice.insert_slice_split(index, ins_slice);
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

    /// Splits the `Rope` at `char_index`, returning the right part of
    /// the split.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_index` is out of bounds (i.e. `char_index > len_chars()`).
    pub fn split_off(&mut self, char_index: usize) -> Self {
        self.try_split_off(char_index).unwrap()
    }

    /// Appends a `Rope` to the end of this one, consuming the other `Rope`.
    ///
    /// Runs in O(log N) time.
    pub fn append(&mut self, mut other: Self) {
        if self.width() == 0 {
            // Special case
            std::mem::swap(self, &mut other);
        } else if other.width() > 0 {
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
    /// - `byte_index` can be one-past-the-end, which will return
    ///   one-past-the-end char index.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_index` is out of bounds (i.e. `byte_index > len_bytes()`).
    #[inline]
    pub fn index_to_width(&self, byte_index: usize) -> usize {
        self.try_index_to_width(byte_index).unwrap()
    }

    /// Returns the byte index of the given char.
    ///
    /// Notes:
    ///
    /// - `char_index` can be one-past-the-end, which will return
    ///   one-past-the-end byte index.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_index` is out of bounds (i.e. `char_index > len_chars()`).
    #[inline]
    pub fn width_to_index(&self, width: usize) -> usize {
        self.try_width_to_index(width).unwrap()
    }

    //-----------------------------------------------------------------------
    // Fetch methods

    /// Returns the byte at `byte_index`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_index` is out of bounds (i.e. `byte_index >= len_bytes()`).
    #[inline]
    pub fn from_index(&self, index: usize) -> M {
        // Bounds check
        if let Some(out) = self.get_from_index(index) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: byte index {}, Rope byte length {}",
                index,
                self.len()
            );
        }
    }

    /// Returns the char at `char_index`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_index` is out of bounds (i.e. `char_index >= len_chars()`).
    #[inline]
    pub fn from_width(&self, width: usize) -> M {
        if let Some(out) = self.get_from_width(width) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: char index {}, Rope char length {}",
                width,
                self.width()
            );
        }
    }

    /// Returns the chunk containing the given byte index.
    ///
    /// Also returns the byte and char indices of the beginning of the chunk
    /// and the index of the line that the chunk starts on.
    ///
    /// Note: for convenience, a one-past-the-end `byte_index` returns the last
    /// chunk of the `RopeSlice`.
    ///
    /// The return value is organized as
    /// `(chunk, chunk_byte_index, chunk_char_index, chunk_line_index)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_index` is out of bounds (i.e. `byte_index > len_bytes()`).
    #[inline]
    pub fn chunk_at_index(&self, index: usize) -> (&[M], usize, usize) {
        // Bounds check
        if let Some(out) = self.get_chunk_at_index(index) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: byte index {}, Rope byte length {}",
                index,
                self.len()
            );
        }
    }

    /// Returns the chunk containing the given char index.
    ///
    /// Also returns the byte and char indices of the beginning of the chunk
    /// and the index of the line that the chunk starts on.
    ///
    /// Note: for convenience, a one-past-the-end `char_index` returns the last
    /// chunk of the `RopeSlice`.
    ///
    /// The return value is organized as
    /// `(chunk, chunk_byte_index, chunk_char_index, chunk_line_index)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_index` is out of bounds (i.e. `char_index > len_chars()`).
    #[inline]
    pub fn chunk_at_width(&self, width: usize) -> (&[M], usize, usize) {
        if let Some(out) = self.get_chunk_at_width(width) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: char index {}, Rope char length {}",
                width,
                self.width()
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
    pub fn width_slice<R>(&self, width_range: R) -> RopeSlice<M>
    where
        R: RangeBounds<usize>,
    {
        self.get_width_slice(width_range).unwrap()
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
    pub fn index_slice<R>(&self, index_range: R) -> RopeSlice<M>
    where
        R: RangeBounds<usize>,
    {
        match self.get_index_slice_impl(index_range) {
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
    /// `byte_index`.
    ///
    /// If `byte_index == len_bytes()` then an iterator at the end of the
    /// `Rope` is created (i.e. `next()` will return `None`).
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_index` is out of bounds (i.e. `byte_index > len_bytes()`).
    #[inline]
    pub fn iter_at_index(&self, index: usize) -> Iter<M> {
        if let Some(out) = self.get_iter_at_index(index) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: byte index {}, Rope byte length {}",
                index,
                self.len()
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
    /// iterator starting at the chunk containing `byte_index`.
    ///
    /// Also returns the byte and char indices of the beginning of the first
    /// chunk to be yielded, and the index of the line that chunk starts on.
    ///
    /// If `byte_index == len_bytes()` an iterator at the end of the `Rope`
    /// (yielding `None` on a call to `next()`) is created.
    ///
    /// The return value is organized as
    /// `(iterator, chunk_byte_index, chunk_char_index, chunk_line_index)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `byte_index` is out of bounds (i.e. `byte_index > len_bytes()`).
    #[inline]
    pub fn chunks_at_index(&self, index: usize) -> (Chunks<M>, usize, usize) {
        if let Some(out) = self.get_chunks_at_index(index) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: byte index {}, Rope byte length {}",
                index,
                self.len()
            );
        }
    }

    /// Creates an iterator over the chunks of the `Rope`, with the
    /// iterator starting at the chunk containing `char_index`.
    ///
    /// Also returns the byte and char indices of the beginning of the first
    /// chunk to be yielded, and the index of the line that chunk starts on.
    ///
    /// If `char_index == len_chars()` an iterator at the end of the `Rope`
    /// (yielding `None` on a call to `next()`) is created.
    ///
    /// The return value is organized as
    /// `(iterator, chunk_byte_index, chunk_char_index, chunk_line_index)`.
    ///
    /// Runs in O(log N) time.
    ///
    /// # Panics
    ///
    /// Panics if `char_index` is out of bounds (i.e. `char_index > len_chars()`).
    #[inline]
    pub fn chunks_at_width(&self, width: usize) -> (Chunks<M>, usize, usize) {
        if let Some(out) = self.get_chunks_at_width(width) {
            out
        } else {
            panic!(
                "Attempt to index past end of Rope: char index {}, Rope char length {}",
                width,
                self.width()
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
    pub fn try_insert(&mut self, char_index: usize, elements: &[M]) -> Result<()> {
        // Bounds check
        if char_index <= self.width() {
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
                let right = self.split_off(char_index);
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
                    let split_index = text.len() - (MAX_BYTES - 4).min(text.len());
                    let ins_text = &text[split_index..];
                    text = &text[..split_index];

                    // Do the insertion.
                    self.insert_internal(char_index, ins_text);
                }
            }
            Ok(())
        } else {
            Err(Error::WidthOutOfBounds(char_index, self.width()))
        }
    }

    /// Non-panicking version of [`insert_char()`](Rope::insert_char).
    #[inline]
    pub fn try_insert_single(&mut self, width: usize, measurable: M) -> Result<()> {
        // Bounds check
        if width <= self.width() {
            self.insert_internal(width, &[measurable]);
            Ok(())
        } else {
            Err(Error::WidthOutOfBounds(width, self.width()))
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
        let end = end_opt.unwrap_or_else(|| self.width());
        if end.max(start) > self.width() {
            Err(Error::WidthRangeOutOfBounds(
                start_opt,
                end_opt,
                self.width(),
            ))
        } else if start > end {
            // A special case that the rest of the logic doesn't handle correctly
            Err(Error::WidthRangeInvalid(start, end))
        } else {
            if start == 0 && end == self.width() {
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
    pub fn try_split_off(&mut self, char_index: usize) -> Result<Self> {
        // Bounds check
        if char_index <= self.width() {
            if char_index == 0 {
                // Special case 1
                let mut new_rope = Rope::new();
                std::mem::swap(self, &mut new_rope);
                Ok(new_rope)
            } else if char_index == self.width() {
                // Special case 2
                Ok(Rope::new())
            } else {
                // Do the split
                let mut new_rope = Rope {
                    root: Arc::new(Arc::make_mut(&mut self.root).split(char_index)),
                };

                // Fix up the edges
                Arc::make_mut(&mut self.root).zip_fix_right();
                Arc::make_mut(&mut new_rope.root).zip_fix_left();
                self.pull_up_singular_nodes();
                new_rope.pull_up_singular_nodes();

                Ok(new_rope)
            }
        } else {
            Err(Error::WidthOutOfBounds(char_index, self.width()))
        }
    }

    /// Non-panicking version of [`byte_to_char()`](Rope::byte_to_char).
    #[inline]
    pub fn try_index_to_width(&self, index: usize) -> Result<usize> {
        // Bounds check
        if index <= self.len() {
            let (chunk, b, c) = self.chunk_at_index(index);
            Ok(c + index_to_width(chunk, index - b))
        } else {
            Err(Error::IndexOutOfBounds(index, self.len()))
        }
    }

    /// Non-panicking version of [`char_to_byte()`](Rope::char_to_byte).
    #[inline]
    pub fn try_width_to_index(&self, width: usize) -> Result<usize> {
        // Bounds check
        if width <= self.width() {
            let (chunk, b, c) = self.chunk_at_width(width);
            Ok(b + width_to_index(chunk, width - c))
        } else {
            Err(Error::WidthOutOfBounds(width, self.width()))
        }
    }

    /// Non-panicking version of [`byte()`](Rope::byte).
    #[inline]
    pub fn get_from_index(&self, index: usize) -> Option<M> {
        // Bounds check
        if index < self.len() {
            let (chunk, chunk_byte_index, _) = self.chunk_at_index(index);
            let chunk_rel_byte_index = index - chunk_byte_index;
            Some(chunk[chunk_rel_byte_index])
        } else {
            None
        }
    }

    /// Non-panicking version of [`char()`](Rope::char).
    #[inline]
    pub fn get_from_width(&self, width: usize) -> Option<M> {
        // Bounds check
        if width < self.width() {
            let (chunk, _, chunk_width) = self.chunk_at_width(width);
            let byte_index = width_to_index(chunk, width - chunk_width);
            Some(chunk[byte_index])
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunk_at_byte()`](Rope::chunk_at_byte).
    #[inline]
    pub fn get_chunk_at_index(&self, byte_index: usize) -> Option<(&[M], usize, usize)> {
        // Bounds check
        if byte_index <= self.len() {
            let (chunk, info) = self.root.get_chunk_at_index(byte_index);
            Some((chunk, info.len as usize, info.width as usize))
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunk_at_char()`](Rope::chunk_at_char).
    #[inline]
    pub fn get_chunk_at_width(&self, width: usize) -> Option<(&[M], usize, usize)> {
        // Bounds check
        if width <= self.width() {
            let (chunk, info) = self.root.get_chunk_at_width(width);
            Some((chunk, info.len as usize, info.width as usize))
        } else {
            None
        }
    }

    /// Non-panicking version of [`slice()`](Rope::slice).
    #[inline]
    pub fn get_width_slice<R>(&self, width_range: R) -> Option<RopeSlice<M>>
    where
        R: RangeBounds<usize>,
    {
        let start = start_bound_to_num(width_range.start_bound()).unwrap_or(0);
        let end = end_bound_to_num(width_range.end_bound()).unwrap_or_else(|| self.width());

        // Bounds check
        if start <= end && end <= self.width() {
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
                    return Err(Error::IndexRangeInvalid(s, e));
                } else if e > self.len() {
                    return Err(Error::IndexRangeOutOfBounds(
                        start_range,
                        end_range,
                        self.len(),
                    ));
                }
            }
            (Some(s), None) => {
                if s > self.len() {
                    return Err(Error::IndexRangeOutOfBounds(
                        start_range,
                        end_range,
                        self.len(),
                    ));
                }
            }
            (None, Some(e)) => {
                if e > self.len() {
                    return Err(Error::IndexRangeOutOfBounds(None, end_range, self.len()));
                }
            }
            _ => {}
        }

        let (start, end) = (
            start_range.unwrap_or(0),
            end_range.unwrap_or_else(|| self.len()),
        );

        RopeSlice::new_with_index_range(&self.root, start, end)
    }

    /// Non-panicking version of [`iter_at()`](Self::iter_at).
    #[inline]
    pub fn get_iter_at_index(&self, index: usize) -> Option<Iter<M>> {
        // Bounds check
        if index <= self.len() {
            let info = self.root.slice_info();
            Some(Iter::new_with_range_at(
                &self.root,
                index,
                (0, info.len as usize),
                (0, info.width as usize),
            ))
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunks_at_byte()`](Rope::chunks_at_byte).
    #[inline]
    pub fn get_chunks_at_index(&self, index: usize) -> Option<(Chunks<M>, usize, usize)> {
        // Bounds check
        if index <= self.len() {
            Some(Chunks::new_with_range_at_index(
                &self.root,
                index,
                (0, self.len()),
                (0, self.width()),
            ))
        } else {
            None
        }
    }

    /// Non-panicking version of [`chunks_at_char()`](Rope::chunks_at_char).
    #[inline]
    pub fn get_chunks_at_width(&self, char_index: usize) -> Option<(Chunks<M>, usize, usize)> {
        // Bounds check
        if char_index <= self.width() {
            Some(Chunks::new_with_range_at_width(
                &self.root,
                char_index,
                (0, self.len()),
                (0, self.width()),
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
        let mut vec = Vec::with_capacity(r.len());
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
        self.width_slice(..) == other.width_slice(..)
    }
}

impl<'a, M> std::cmp::PartialEq<&'a [M]> for Rope<M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &&'a [M]) -> bool {
        self.width_slice(..) == *other
    }
}

impl<'a, M> std::cmp::PartialEq<Rope<M>> for &'a [M]
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        *self == other.width_slice(..)
    }
}

impl<M> std::cmp::PartialEq<[M]> for Rope<M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &[M]) -> bool {
        self.width_slice(..) == other
    }
}

impl<M> std::cmp::PartialEq<Rope<M>> for [M]
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        self == other.width_slice(..)
    }
}

impl<'a, M> std::cmp::PartialEq<Vec<M>> for Rope<M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Vec<M>) -> bool {
        self.width_slice(..) == other.as_slice()
    }
}

impl<'a, M> std::cmp::PartialEq<Rope<M>> for Vec<M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        self.as_slice() == other.width_slice(..)
    }
}

impl<'a, M> std::cmp::PartialEq<std::borrow::Cow<'a, [M]>> for Rope<M>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &std::borrow::Cow<'a, [M]>) -> bool {
        self.width_slice(..) == **other
    }
}

impl<'a, M> std::cmp::PartialEq<Rope<M>> for std::borrow::Cow<'a, [M]>
where
    M: Measurable + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Rope<M>) -> bool {
        **self == other.width_slice(..)
    }
}

impl<M> std::cmp::Ord for Rope<M>
where
    M: Measurable + Ord,
{
    #[inline]
    fn cmp(&self, other: &Rope<M>) -> std::cmp::Ordering {
        self.width_slice(..).cmp(&other.width_slice(..))
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

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Lipsum {
        Lorem,
        Ipsum,
        Dolor(usize),
        Sit,
        Amet,
        Consectur(&'static str),
        Adipiscing(bool),
    }

    impl Measurable for Lipsum {
        fn width(&self) -> usize {
            match self {
                Lipsum::Lorem => 1,
                Lipsum::Ipsum => 2,
                Lipsum::Dolor(width) => *width,
                Lipsum::Sit => 0,
                Lipsum::Amet => 0,
                Lipsum::Consectur(text) => text.len(),
                Lipsum::Adipiscing(boolean) => *boolean as usize,
            }
        }
    }

    use self::Lipsum::*;
    /// 70 elements, total width of 135.
    fn lorem_ipsum() -> Vec<Lipsum> {
        (0..70)
            .into_iter()
            .map(|num| match num % 14 {
                0 | 7 => Lorem,
                1 | 8 => Ipsum,
                2 => Dolor(4),
                3 | 10 => Sit,
                4 | 11 => Amet,
                5 => Consectur("hello"),
                6 => Adipiscing(true),
                9 => Dolor(8),
                12 => Consectur("bye"),
                13 => Adipiscing(false),
                _ => unreachable!(),
            })
            .collect()
    }

    /// 5 elements, total width of 6.
    const SHORT_LOREM: &[Lipsum] = &[Lorem, Ipsum, Dolor(3), Sit, Amet];

    #[test]
    fn new_01() {
        let rope: Rope<Lipsum> = Rope::new();
        assert_eq!(rope, [].as_slice());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn from_slice() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        assert_eq!(rope, lorem_ipsum());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn len_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        assert_eq!(rope.len(), 70);
    }

    #[test]
    fn width_02() {
        let rope: Rope<Lipsum> = Rope::from_slice(&[]);
        assert_eq!(rope.len(), 0);
    }

    #[test]
    fn len_from_widths_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        assert_eq!(rope.width(), 135);
    }

    #[test]
    fn len_from_widths_02() {
        let rope: Rope<Lipsum> = Rope::from_slice(&[]);
        assert_eq!(rope.width(), 0);
    }

    #[test]
    fn insert_01() {
        let mut rope = Rope::from_slice(SHORT_LOREM);
        rope.insert_slice(3, &[Lorem, Ipsum, Dolor(3)]);

        assert_eq!(
            rope,
            [Lorem, Ipsum, Lorem, Ipsum, Dolor(3), Dolor(3), Sit, Amet].as_slice()
        );

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn insert_02() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());
        rope.insert_slice(0, &[Lorem, Ipsum, Dolor(3)]);

        assert_eq!(
            rope,
            [Lorem, Ipsum, Dolor(3), Lorem, Ipsum, Dolor(3), Sit, Amet].as_slice()
        );

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn insert_03() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());
        rope.insert_slice(5, &[Lorem, Ipsum, Dolor(3)]);

        assert_eq!(
            rope,
            [Lorem, Ipsum, Dolor(3), Sit, Amet, Lorem, Ipsum, Dolor(3)].as_slice()
        );

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn insert_04() {
        let mut rope = Rope::new();
        rope.insert_slice(0, &[Lorem, Ipsum]);
        rope.insert_slice(2, &[Dolor(5)]);
        rope.insert_slice(3, &[Sit]);
        rope.insert_slice(4, &[Consectur("test")]);
        rope.insert_slice(11, &[Dolor(3)]);

        // NOTE: Inserting in the middle of an item'slice width range, makes it so
        // you actually place it at the end of said item.
        assert_eq!(
            rope,
            [Lorem, Dolor(5), Consectur("test"), Sit, Ipsum, Dolor(3)].as_slice()
        );

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn insert_05() {
        let mut rope = Rope::new();
        rope.insert_slice(0, &[Dolor(15), Dolor(20)]);
        rope.insert_slice(7, &[Sit, Amet]);
        assert_eq!(rope, [Dolor(15), Sit, Amet, Dolor(20)].as_slice());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn insert_06() {
        let mut rope = Rope::new();
        rope.insert(0, Dolor(15));
        rope.insert(1, Dolor(20));
        rope.insert(2, Dolor(10));
        rope.insert(3, Dolor(4));
        rope.insert_slice(20, &[Sit, Amet]);
        assert_eq!(
            rope,
            [Dolor(4), Dolor(10), Dolor(20), Sit, Amet, Dolor(15)].as_slice()
        );

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn remove_01() {
        let slice = &[Dolor(15), Sit, Amet, Dolor(24), Lorem, Ipsum, Dolor(7)];
        let mut rope = Rope::from_slice(slice);

        rope.remove(5..11); // Removes Dolor(15).
        rope.remove(24..31); // Removes [Lorem, Ipsum, Dolor(7).
        rope.remove(19..25); // Removes Dolor(24).
        assert_eq!(rope, [Sit, Amet].as_slice());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn remove_02() {
        let slice = &[Lorem; 15];
        let mut rope = Rope::from_slice(slice);

        // Make sure CRLF pairs get merged properly, via
        // assert_invariants() below.
        rope.remove(3..6);
        assert_eq!(rope, [Lorem; 12].as_slice());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn remove_03() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());

        // Make sure removing nothing actually does nothing.
        rope.remove(45..45);
        assert_eq!(rope, lorem_ipsum());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn remove_04() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());

        // Make sure removing everything works.
        rope.remove(0..135);
        assert_eq!(rope, [].as_slice());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn remove_05() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());

        // Make sure removing a large range works.
        rope.remove(3..100);
        assert_eq!(rope, &lorem_ipsum()[2..51]);

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn remove_06() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());
        #[allow(clippy::reversed_empty_ranges)]
        rope.remove(56..55); // Wrong ordering of start/end on purpose.
    }

    #[test]
    #[should_panic]
    fn remove_07() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());
        rope.remove(134..136); // Removing past the end
    }

    #[test]
    fn split_off_01() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());

        let split = rope.split_off(50);
        assert_eq!(rope, &lorem_ipsum()[..23]);
        assert_eq!(split, &lorem_ipsum()[23..]);

        rope.assert_integrity();
        split.assert_integrity();
        rope.assert_invariants();
        split.assert_invariants();
    }

    #[test]
    fn split_off_02() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());

        let split = rope.split_off(1);
        assert_eq!(rope, [Lorem].as_slice());
        assert_eq!(split, &lorem_ipsum()[1..]);

        rope.assert_integrity();
        split.assert_integrity();
        rope.assert_invariants();
        split.assert_invariants();
    }

    #[test]
    fn split_off_03() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());

        let split = rope.split_off(134);
        assert_eq!(rope, &lorem_ipsum()[..69]);
        assert_eq!(split, [Consectur("bye"), Adipiscing(false)].as_slice());

        rope.assert_integrity();
        split.assert_integrity();
        rope.assert_invariants();
        split.assert_invariants();
    }

    #[test]
    fn split_off_04() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());

        let split = rope.split_off(0);
        assert_eq!(rope, [].as_slice());
        assert_eq!(split, lorem_ipsum().as_slice());

        rope.assert_integrity();
        split.assert_integrity();
        rope.assert_invariants();
        split.assert_invariants();
    }

    #[test]
    fn split_off_05() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());

        let split = rope.split_off(135);
        assert_eq!(rope, lorem_ipsum().as_slice());
        assert_eq!(split, [].as_slice());

        rope.assert_integrity();
        split.assert_integrity();
        rope.assert_invariants();
        split.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn split_off_06() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());
        rope.split_off(104); // One past the end of the rope
    }

    #[test]
    fn append_01() {
        let mut rope = Rope::from_slice(&lorem_ipsum()[..35]);
        let append = Rope::from_slice(&lorem_ipsum()[35..]);

        rope.append(append);
        assert_eq!(rope, lorem_ipsum().as_slice());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn append_02() {
        let mut rope = Rope::from_slice(&lorem_ipsum()[..68]);
        let append = Rope::from_slice(&[Consectur("bye"), Adipiscing(false)]);

        rope.append(append);
        assert_eq!(rope, lorem_ipsum());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn append_03() {
        let mut rope = Rope::from_slice(&[Lorem, Ipsum]);
        let append = Rope::from_slice(&lorem_ipsum()[2..]);

        rope.append(append);
        assert_eq!(rope, lorem_ipsum());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn append_04() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());
        let append = Rope::from_slice([].as_slice());

        rope.append(append);
        assert_eq!(rope, lorem_ipsum());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn append_05() {
        let mut rope = Rope::from_slice([].as_slice());
        let append = Rope::from_slice(lorem_ipsum().as_slice());

        rope.append(append);
        assert_eq!(rope, lorem_ipsum());

        rope.assert_integrity();
        rope.assert_invariants();
    }

    #[test]
    fn width_to_index_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        assert_eq!(rope.width_to_index(0), 0);
        assert_eq!(rope.width_to_index(1), 1);
        assert_eq!(rope.width_to_index(2), 2);

        assert_eq!(rope.width_to_index(91), 48);
        assert_eq!(rope.width_to_index(92), 48);
        assert_eq!(rope.width_to_index(93), 49);
        assert_eq!(rope.width_to_index(94), 50);

        assert_eq!(rope.width_to_index(102), 52);
        assert_eq!(rope.width_to_index(103), 52);
    }

    #[test]
    fn from_index_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        assert_eq!(rope.from_index(0), Lorem);

        // UTF-8 for "wide exclamation mark"
        assert_eq!(rope.from_index(67), Amet);
        assert_eq!(rope.from_index(68), Consectur("bye"));
        assert_eq!(rope.from_index(69), Adipiscing(false));
    }

    #[test]
    #[should_panic]
    fn from_index_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        rope.from_index(70);
    }

    #[test]
    #[should_panic]
    fn from_index_03() {
        let rope: Rope<Lipsum> = Rope::from_slice(&[]);
        rope.from_index(0);
    }

    #[test]
    fn from_width_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        assert_eq!(rope.from_width(0), Lorem);
        assert_eq!(rope.from_width(10), Consectur("hello"));
        assert_eq!(rope.from_width(18), Dolor(8));
        assert_eq!(rope.from_width(108), Adipiscing(false));
    }

    #[test]
    #[should_panic]
    fn from_width_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        rope.from_width(103);
    }

    #[test]
    #[should_panic]
    fn from_width_03() {
        let rope: Rope<Lipsum> = Rope::from_slice(&[]);
        rope.from_width(0);
    }

    #[test]
    fn chunk_at_index() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let lorem_ipsum = lorem_ipsum();
        let mut total = lorem_ipsum.as_slice();

        let mut last_chunk = [].as_slice();
        for i in 0..rope.len() {
            let (chunk, b, c) = rope.chunk_at_index(i);
            assert_eq!(c, index_to_width(&total, b));
            if chunk != last_chunk {
                assert_eq!(chunk, &total[..chunk.len()]);
                total = &total[chunk.len()..];
                last_chunk = chunk;
            }

            let c1 = {
                let i2 = index_to_width(&lorem_ipsum, i);
                lorem_ipsum.iter().nth(i2).unwrap()
            };
            let c2 = {
                let i2 = i - b;
                let i3 = index_to_width(chunk, i2);
                chunk.iter().nth(i3).unwrap()
            };
            assert_eq!(c1, c2);
        }
        assert_eq!(total.len(), 0);
    }

    #[test]
    fn chunk_at_width_asdf() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let lorem_ipsum = lorem_ipsum();
        let mut total = lorem_ipsum.as_slice();

        let mut last_chunk = [].as_slice();
        for i in 0..rope.width() {
            let (chunk, index, width) = rope.chunk_at_width(i);
            if chunk != last_chunk {
                assert_eq!(chunk, &total[..chunk.len()]);
                total = &total[chunk.len()..];
                last_chunk = chunk;
            }

            let index_1 = width_to_index(&lorem_ipsum, i);
            let lipsum_1 = lorem_ipsum.iter().nth(index_1).unwrap();
            let index_2 = width_to_index(&chunk, i - width);
            let lipsum_2 = chunk.iter().nth(index_2).unwrap();
            assert_eq!(lipsum_1, lipsum_2);
        }
        assert_eq!(total.len(), 0);
    }

    #[test]
    fn width_slice_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        let slice = rope.width_slice(0..rope.width());

        assert_eq!(lorem_ipsum(), slice);
    }

    #[test]
    fn width_slice_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        let slice = rope.width_slice(5..21);

        assert_eq!(&lorem_ipsum()[2..9], slice);
    }

    #[test]
    fn width_slice_03() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        let slice = rope.width_slice(31..135);

        assert_eq!(&lorem_ipsum()[17..70], slice);
    }

    #[test]
    fn width_slice_04() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        let slice = rope.width_slice(53..53);

        assert_eq!([].as_slice(), slice);
    }

    #[test]
    #[should_panic]
    fn width_slice_05() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        #[allow(clippy::reversed_empty_ranges)]
        rope.width_slice(53..52); // Wrong ordering on purpose.
    }

    #[test]
    #[should_panic]
    fn width_slice_06() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        rope.width_slice(134..136);
    }

    #[test]
    fn index_slice_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        let slice = rope.index_slice(0..rope.len());

        assert_eq!(lorem_ipsum(), slice);
    }

    #[test]
    fn index_slice_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        let slice = rope.index_slice(5..21);

        assert_eq!(&lorem_ipsum()[5..21], slice);
    }

    #[test]
    fn index_slice_03() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        let slice = rope.index_slice(31..55);

        assert_eq!(&lorem_ipsum()[31..55], slice);
    }

    #[test]
    fn index_slice_04() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        let slice = rope.index_slice(53..53);

        assert_eq!([].as_slice(), slice);
    }

    #[test]
    #[should_panic]
    fn index_slice_05() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        #[allow(clippy::reversed_empty_ranges)]
        rope.index_slice(53..52); // Wrong ordering on purpose.
    }

    #[test]
    #[should_panic]
    fn index_slice_06() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        rope.index_slice(20..72);
    }

    #[test]
    fn eq_rope_01() {
        let rope: Rope<Lipsum> = Rope::from_slice([].as_slice());

        assert_eq!(rope, rope);
    }

    #[test]
    fn eq_rope_02() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        assert_eq!(rope, rope);
    }

    #[test]
    fn eq_rope_03() {
        let rope_1 = Rope::from_slice(lorem_ipsum().as_slice());
        let mut rope_2 = rope_1.clone();
        rope_2.remove(26..27);
        rope_2.insert(26, Consectur("bye"));

        assert_ne!(rope_1, rope_2);
    }

    #[test]
    fn eq_rope_04() {
        let rope: Rope<Lipsum> = Rope::from_slice([].as_slice());

        assert_eq!(rope, [].as_slice());
        assert_eq!([].as_slice(), rope);
    }

    #[test]
    fn eq_rope_05() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());

        assert_eq!(rope, lorem_ipsum());
        assert_eq!(lorem_ipsum(), rope);
    }

    #[test]
    fn eq_rope_06() {
        let mut rope = Rope::from_slice(lorem_ipsum().as_slice());
        rope.remove(26..27);
        rope.insert(26, Consectur("bye"));

        assert_ne!(rope, lorem_ipsum());
        assert_ne!(lorem_ipsum(), rope);
    }

    #[test]
    fn eq_rope_07() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice: Vec<Lipsum> = lorem_ipsum().into();

        assert_eq!(rope, slice);
        assert_eq!(slice, rope);
    }

    #[test]
    fn to_string_01() {
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let slice: Vec<Lipsum> = (&rope).into();

        assert_eq!(rope, slice);
    }

    #[test]
    fn to_cow_01() {
        use std::borrow::Cow;
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let cow: Cow<[Lipsum]> = (&rope).into();

        assert_eq!(rope, cow);
    }

    #[test]
    fn to_cow_02() {
        use std::borrow::Cow;
        let rope = Rope::from_slice(lorem_ipsum().as_slice());
        let cow: Cow<[Lipsum]> = (rope.clone()).into();

        assert_eq!(rope, cow);
    }

    #[test]
    fn to_cow_03() {
        use std::borrow::Cow;
        let rope = Rope::from_slice(&[Lorem]);
        let cow: Cow<[Lipsum]> = (&rope).into();

        // Make sure it'slice borrowed.
        if let Cow::Owned(_) = cow {
            panic!("Small Cow conversions should result in a borrow.");
        }

        assert_eq!(rope, cow);
    }

    #[test]
    fn from_rope_slice_01() {
        let rope_1 = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope_1.width_slice(..);
        let rope_2: Rope<Lipsum> = slice.into();

        assert_eq!(rope_1, rope_2);
        assert_eq!(slice, rope_2);
    }

    #[test]
    fn from_rope_slice_02() {
        let rope_1 = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope_1.width_slice(0..24);
        let rope_2: Rope<Lipsum> = slice.into();

        assert_eq!(slice, rope_2);
    }

    #[test]
    fn from_rope_slice_03() {
        let rope_1 = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope_1.width_slice(13..89);
        let rope_2: Rope<Lipsum> = slice.into();

        assert_eq!(slice, rope_2);
    }

    #[test]
    fn from_rope_slice_04() {
        let rope_1 = Rope::from_slice(lorem_ipsum().as_slice());
        let slice = rope_1.width_slice(13..41);
        let rope_2: Rope<Lipsum> = slice.into();

        assert_eq!(slice, rope_2);
    }

    #[test]
    fn from_iter_01() {
        let rope_1 = Rope::from_slice(lorem_ipsum().as_slice());
        let rope_2 = Rope::from_iter(rope_1.chunks());

        assert_eq!(rope_1, rope_2);
    }

    #[test]
    fn is_instance_01() {
        let rope = Rope::from_slice(&[Lorem, Ipsum, Dolor(10), Sit, Amet]);
        let mut c1 = rope.clone();
        let c2 = c1.clone();

        assert!(rope.is_instance(&c1));
        assert!(rope.is_instance(&c2));
        assert!(c1.is_instance(&c2));

        c1.insert_slice(0, &[Consectur("oh noes!")]);

        assert!(!rope.is_instance(&c1));
        assert!(rope.is_instance(&c2));
        assert!(!c1.is_instance(&c2));
    }

    // Iterator tests are in the iter module
}
