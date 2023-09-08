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
    fn measure(&self) -> usize {
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
#[test]
#[cfg_attr(miri, ignore)]
fn from_str() {
    let mut rng = rand::thread_rng();

    let small_vec: Vec<Lipsum> = {
        (0..1000)
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
    // Build rope from file contents
    let rope = Rope::from_slice(small_vec.as_slice());

    // Verify rope integrity
    rope.assert_integrity();
    rope.assert_invariants();

    // Verify that they match
    assert_eq!(rope, small_vec.as_slice());
}
