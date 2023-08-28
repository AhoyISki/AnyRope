use any_rope::{Measurable, Rope};
use rand::Rng;

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

#[test]
#[cfg_attr(miri, ignore)]
fn remove_at_chunk_boundery() {
    let mut rng = rand::thread_rng();

    let medium_vec: Vec<Lipsum> = {
        (0..100000)
            .map(|_| match rng.gen::<usize>() % 14 {
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
    };

    let mut r = Rope::from_slice(medium_vec.as_slice());
    // remove exactly at a chunk boundry
    // to trigger an edgecase in fix_tree_seam
    r.remove_exclusive(31354..58881);

    // Verify rope integrity
    r.assert_integrity();
    r.assert_invariants();
}
