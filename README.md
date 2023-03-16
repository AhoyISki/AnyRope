# AnyRope

[![CI Build Status][github-ci-img]][github-ci]
[![Latest Release][crates-io-badge]][crates-io-url]
[![Documentation][docs-rs-img]][docs-rs-url]

AnyRope is an arbitrary data type rope for Rust, designed for similar operations
that a rope would do, but targeted at data types that are not text.


## Example Usage

An example of where this would be useful is in the tagging of text in a text editor,
for example, one may assossiate a rope of text with a rope of tags.

```rust
// The tags that will be assossiated with a piece of text, that could be a rope.
struct Tag {
    PrintRed,
    Underline,
    Normal,
    Skip(usize)
}

use Tag::*;
impl any_rope::Measurable for Tag {
    fn width(&self) -> usize {
        match self {
            // The zero here represents the fact that multiple tags may be placed
            // in the same character.
            PrintRed | Underline | Normal => 0,
            // Skip here is an amount of characters with no tags in them.
            Skip(amount) => *amount
        }
    }
}

// An `&str` that will be colored.
let my_str = "This word will be red!";

// Here's what this means:
// - Skip 5 characters;
// - Change the color to red;
// - Skip 4 characters;
// - Change the rendering back to normal.
let mut tags = any_rope::Rope::from_slice(&[Skip(5), PrintRed, Skip(4), Normal]);
// Do note that Tag::Skip only represents characters because we are also iterating
// over a `Chars` iterator, and have chosen to do so.

// An empty range, when used in an inclusive removal will remove all
// 0 width elements in that specific width.
// `Rope::remove_exclusive()` would keep them.
// In this case, that would be `Tag::PrintRed`
tags.remove_inclusive(5..5);
// In place of that `Tag::PrintRed`, we will insert `Tag::Underline`.
tags.insert(5, Underline);

// The AnyRope iterator not only returns the element in question, but also the width
// where it starts.
let mut tags_iter = tags.iter().peekable();

for (cur_index, ch) in my_str.chars().enumerate() {
    // The while let loop here is a useful way to activate all tags within the same
    // character. For example, we could have a sequence of [Tag::UnderLine, Tag::PrintRed]
    // in the `Rope`, both of which have a width of 0, allowing one to execute multiple
    // `Tag`s in a single character.
    while let Some((index, tag)) = tags_iter.peek() {
        if *index == cur_index {
            activate_tag(tag);
            tags_iter.next();
        } else {
            break;
        }
    }
    print!("{}", ch);
}
```

## When Should I Use AnyRope?

So far, I haven't found a use for AnyRope, other than text editors, but I'm not
discounting the possibility that it may be useful somewhere else.

## Features

### Concept of widths

The width of the element that implements Measurable can be whatever the end user wants
it to be, allowing for great flexibility in how AnyRope could be useful.

### Rope slices

AnyRope has rope slices that allow you to work with just parts of a rope, using
all the read-only operations of a full rope including iterators and making
sub-slices.

### Flexible APIs with low-level access

Although AnyRope is intentionally limited in scope, it also provides APIs for
efficiently accessing and working with its internal slice chunk
representation, allowing additional functionality to be efficiently
implemented by client code with minimal overhead.

### Thread safe

AnyRope ensures that even though clones share memory, everything is thread-safe.
Clones can be sent to other threads for both reading and writing.

## Unsafe code

AnyRope uses unsafe code to help achieve some of its space and performance
characteristics. Although effort has been put into keeping the unsafe code
compartmentalized and making it correct, please be cautious about using AnyRope
in software that may face adversarial conditions.

## License

AnyRope is licensed under the MIT license (LICENSE.md or http://opensource.org/licenses/MIT)

## Contributing

Contributions are absolutely welcome!  However, please open an issue or email
me to discuss larger changes, to avoid doing a lot of work that may get
rejected.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in AnyRope by you will be licensed as above,
without any additional terms or conditions.

[crates-io-badge]: https://img.shields.io/crates/v/any-rope.svg
[crates-io-url]:   https://crates.io/crates/any-rope
[github-ci-img]:   https://github.com/AhoyISki/AnyRope/actions/workflows/ci.yml/badge.svg
[github-ci]:       https://github.com/AhoyISki/AnyRope/actions/workflows/ci.yml
[docs-rs-img]:     https://docs.rs/any-rope/badge.svg
[docs-rs-url]:     https://docs.rs/any-rope
