#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
//! Example of basic search-and-replace functionality implemented on top
//! of AnyRope.
//!
//! The file contents with the search-and-replace performed on it is sent to
//! stdout.

#![allow(clippy::redundant_field_names)]
use any_rope::{iter::Iter, max_children, max_len, Measurable, Rope, RopeSlice};

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

fn main() {
    // Load file contents into a rope.
    let mut text = Rope::from_slice(&[
        Lorem,
        Ipsum,
        Dolor(5),
        Sit,
        Amet,
        Consectur("test"),
        Adipiscing(false),
    ]);

    // Do the search-and-replace.
    search_and_replace(&mut text, &[Dolor(5), Sit, Amet], &[Lorem, Ipsum, Lorem]);

    // Print the new text to stdout.
    println!("{:?}", text);
}

/// Searches the rope for `search_pattern` and replaces all matches with
/// `replacement_text`.
///
/// There are several ways this could be done:  
///
/// 1. Clone the rope and then do the search on the original while replacing on
///    the clone.  This isn't as awful as it sounds because the clone operation
///    is constant-time and the two ropes will share most of their storage in
///    typical cases.  However, this probably isn't the best general solution
///    because it will use a lot of additional space if a large percentage of
///    the text is being replaced.
///
/// 2. A two-stage approach: first find and collect all the matches, then do the
///    replacements on the original rope.  This is a good solution when a
///    relatively small number of matches are expected.  However, if there are a
///    large number of matches then the space to store the matches themselves
///    can become large.
///
/// 3. A piece-meal approach: search for the first match, replace it, then
///    restart the search from there, repeat.  This is a good solution for
///    memory-constrained situations.  However, computationally it is likely the
///    most expensive when there are a large number of matches and there are
///    costs associated with repeatedly restarting the search.
///
/// 4. Combine approaches #2 and #3: collect a fixed number of matches and
///    replace them, then collect another batch of matches and replace them, and
///    so on.  This is probably the best general solution, because it combines
///    the best of both #2 and #3: it allows you to collect the matches in a
///    bounded amount of space, and any costs associated with restarting the
///    search are amortized over multiple matches.
///
/// In this implementation we take approach #4 because it seems the
/// all-around best.
fn search_and_replace<M>(rope: &mut Rope<M>, search_pattern: &[M], replacement_slice: &[M])
where
    M: Measurable + PartialEq,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    const BATCH_SIZE: usize = 256;
    let replacement_text_len = replacement_slice.len();

    let mut head = 0; // Keep track of where we are between searches
    let mut matches = Vec::with_capacity(BATCH_SIZE);
    loop {
        // Collect the next batch of matches.  Note that we don't use
        // `Iterator::collect()` to collect the batch because we want to
        // re-use the same Vec to avoid unnecessary allocations.
        matches.clear();
        for m in
            SearchIter::from_rope_slice(&rope.width_slice(head..), search_pattern).take(BATCH_SIZE)
        {
            matches.push(m);
        }

        // If there are no matches, we're done!
        if matches.is_empty() {
            break;
        }

        // Replace the collected matches.
        let mut index_diff: isize = 0;
        for &(start, end) in matches.iter() {
            // Get the properly offset indices.
            let start_d = (head as isize + start as isize + index_diff) as usize;
            let end_d = (head as isize + end as isize + index_diff) as usize;

            // Do the replacement.
            rope.remove_exclusive(start_d..end_d);
            rope.insert_slice(start_d, replacement_slice);

            // Update the index offset.
            let match_len = (end - start) as isize;
            index_diff = index_diff - match_len + replacement_text_len as isize;
        }

        // Update head for next iteration.
        head = (head as isize + index_diff + matches.last().unwrap().1 as isize) as usize;
    }
}

/// An iterator over simple textual matches in a RopeSlice.
///
/// This implementation is somewhat naive, and could be sped up by using a
/// more sophisticated text searching algorithm such as Boyer-Moore or
/// Knuth-Morris-Pratt.
///
/// The important thing, however, is the interface.  For example, a regex
/// implementation providing an equivalent interface could easily be dropped
/// in, and the search-and-replace function above would work with it quite
/// happily.
struct SearchIter<'a, M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    iter: Iter<'a, M>,
    search_pattern: &'a [M],
    search_pattern_char_len: usize,
    cur_index: usize, // The current char index of the search head.
    possible_matches: Vec<std::slice::Iter<'a, M>>, /* Tracks where we are in the search pattern
                                                     * for the current possible matches. */
}

impl<'a, M> SearchIter<'a, M>
where
    M: Measurable,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    fn from_rope_slice<'b: 'a>(slice: &'b RopeSlice<M>, search_pattern: &'b [M]) -> Self {
        assert!(
            !search_pattern.is_empty(),
            "Can't search using an empty search pattern."
        );
        SearchIter {
            iter: slice.iter(),
            search_pattern,
            search_pattern_char_len: search_pattern.len(),
            cur_index: 0,
            possible_matches: Vec::new(),
        }
    }
}

impl<'a, M> Iterator for SearchIter<'a, M>
where
    M: Measurable + PartialEq,
    [(); max_len::<M>()]: Sized,
    [(); max_children::<M>()]: Sized,
{
    type Item = (usize, usize);

    // Return the start/end char indices of the next match.
    fn next(&mut self) -> Option<(usize, usize)> {
        #[allow(clippy::while_let_on_iterator)]
        while let Some((_, next_element)) = self.iter.next() {
            self.cur_index += 1;

            // Push new potential match, for a possible match starting at the
            // current char.
            self.possible_matches.push(self.search_pattern.iter());

            // Check the rope's char against the next character in each of
            // the potential matches, removing the potential matches that
            // don't match.  We're using indexing instead of iteration here
            // so that we can remove the possible matches as we go.
            let mut i = 0;
            while i < self.possible_matches.len() {
                let pattern_char = self.possible_matches[i].next().unwrap();
                if next_element == *pattern_char {
                    if self.possible_matches[i].clone().next().is_none() {
                        // We have a match!  Reset possible matches and
                        // return the successful match's char indices.
                        let char_match_range = (
                            self.cur_index - self.search_pattern_char_len,
                            self.cur_index,
                        );
                        self.possible_matches.clear();
                        return Some(char_match_range);
                    } else {
                        // Match isn't complete yet, move on to the next.
                        i += 1;
                    }
                } else {
                    // Doesn't match, remove it.
                    let _ = self.possible_matches.swap_remove(i);
                }
            }
        }

        None
    }
}
