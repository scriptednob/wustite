//! A set of squares packed into one `u64`.
//!
//! A [`Bitboard`] has bit `n` set when square `n` (see [`Square`]) is a member of
//! the set. The engine represents everything set-like with it: every piece type,
//! both colours, attack sets, pin rays, move targets. Set algebra becomes single
//! CPU instructions: `&` is intersection, `|` is union, `^` is symmetric
//! difference and `!` is complement.
//!
//! ```text
//!     a b c d e f g h
//!  8  56 57 58 59 60 61 62 63      bit 63 is h8
//!  7  48 49 50 51 52 53 54 55
//!  6  40 41 42 43 44 45 46 47
//!  5  32 33 34 35 36 37 38 39
//!  4  24 25 26 27 28 29 30 31
//!  3  16 17 18 19 20 21 22 23
//!  2   8  9 10 11 12 13 14 15
//!  1   0  1  2  3  4  5  6  7      bit 0 is a1
//! ```
//!
//! # Iterating
//!
//! A bitboard is an [`IntoIterator`] over the squares it contains, lowest bit
//! first. Iteration uses `trailing_zeros` and clears the lowest set bit each step
//! (`x &= x - 1`), so it costs time proportional to the number of set bits rather
//! than 64.
//!
//! ```
//! use wustite::{Bitboard, Square};
//!
//! let set = Square::A1.bb() | Square::C3.bb() | Square::H8.bb();
//! let squares: Vec<Square> = set.into_iter().collect();
//! assert_eq!(squares, [Square::A1, Square::C3, Square::H8]);
//! assert_eq!(set.count(), 3);
//! ```
//!
//! # Shifts
//!
//! [`Bitboard::north`], [`Bitboard::east`] and friends shift every square at once.
//! Horizontal and diagonal shifts mask off the file that would wrap around the
//! edge of the board, so pushing every pawn one square is one shift and one mask.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use crate::types::Square;

/// A set of squares stored as the bits of a `u64`. See the [module documentation](crate::bitboard).
///
/// `Debug` formatting prints an 8x8 diagram with rank 8 at the top, which is handy
/// in tests and when debugging move generation.
///
/// ```
/// use wustite::{Bitboard, Square};
///
/// let a = Square::E4.bb() | Square::E5.bb();
/// let b = Square::E5.bb() | Square::E6.bb();
/// assert_eq!(a & b, Square::E5.bb());
/// assert_eq!((a | b).count(), 3);
/// assert!(Bitboard::EMPTY.is_empty());
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Bitboard(pub u64);

impl Bitboard {
    /// The empty set.
    pub const EMPTY: Bitboard = Bitboard(0);
    /// All 64 squares.
    pub const FULL: Bitboard = Bitboard(!0);

    /// Every square on the a-file.
    pub const FILE_A: Bitboard = Bitboard(0x0101_0101_0101_0101);
    /// Every square on the h-file.
    pub const FILE_H: Bitboard = Bitboard(0x8080_8080_8080_8080);

    /// Every square on the first rank.
    pub const RANK_1: Bitboard = Bitboard(0x0000_0000_0000_00FF);
    /// Every square on the second rank.
    pub const RANK_2: Bitboard = Bitboard(0x0000_0000_0000_FF00);
    /// Every square on the third rank. White pawns that single-push from the second rank land here, which move generation uses to find double pushes.
    pub const RANK_3: Bitboard = Bitboard(0x0000_0000_00FF_0000);
    /// Every square on the sixth rank. The black counterpart of [`Bitboard::RANK_3`].
    pub const RANK_6: Bitboard = Bitboard(0x0000_FF00_0000_0000);
    /// Every square on the seventh rank.
    pub const RANK_7: Bitboard = Bitboard(0x00FF_0000_0000_0000);
    /// Every square on the eighth rank. White promotes here.
    pub const RANK_8: Bitboard = Bitboard(0xFF00_0000_0000_0000);

    /// Whether the set has no squares.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether the set has at least one square. The negation of [`Bitboard::is_empty`], written this way where it reads better in conditions.
    #[inline(always)]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    /// The number of squares in the set (population count).
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Whether the set has two or more squares.
    ///
    /// This is cheaper than `count() > 1`: `x & (x - 1)` clears the lowest bit, and
    /// the result is non-zero exactly when another bit remained. Move generation uses
    /// it for double check and for detecting a single blocker between a slider and
    /// the king.
    #[inline(always)]
    pub const fn more_than_one(self) -> bool {
        self.0 & self.0.wrapping_sub(1) != 0
    }

    /// Whether `sq` is in the set.
    #[inline(always)]
    pub const fn contains(self, sq: Square) -> bool {
        (self.0 >> sq.0) & 1 != 0
    }

    /// The lowest square in the set.
    ///
    /// The result is meaningless for an empty bitboard (it returns index 64), so
    /// check [`Bitboard::any`] first when the set might be empty.
    #[inline(always)]
    pub const fn lsb(self) -> Square {
        Square(self.0.trailing_zeros() as u8)
    }

    /// Removes and returns the lowest square in the set. The set must not be empty.
    #[inline(always)]
    pub fn pop_lsb(&mut self) -> Square {
        let sq = self.lsb();
        self.0 &= self.0.wrapping_sub(1);
        sq
    }

    /// Shifts every square one rank toward rank 8. Squares on rank 8 fall off the board.
    #[inline(always)]
    pub const fn north(self) -> Bitboard {
        Bitboard(self.0 << 8)
    }

    /// Shifts every square one rank toward rank 1. Squares on rank 1 fall off the board.
    #[inline(always)]
    pub const fn south(self) -> Bitboard {
        Bitboard(self.0 >> 8)
    }

    /// Shifts every square one file toward the h-file. Squares on the h-file are dropped instead of wrapping to the next rank.
    #[inline(always)]
    pub const fn east(self) -> Bitboard {
        Bitboard((self.0 & !Self::FILE_H.0) << 1)
    }

    /// Shifts every square one file toward the a-file. Squares on the a-file are dropped instead of wrapping to the previous rank.
    #[inline(always)]
    pub const fn west(self) -> Bitboard {
        Bitboard((self.0 & !Self::FILE_A.0) >> 1)
    }

    /// Shifts every square one step diagonally toward h8, dropping squares that would leave the board. White pawn captures toward the h-file use this shift.
    #[inline(always)]
    pub const fn north_east(self) -> Bitboard {
        Bitboard((self.0 & !Self::FILE_H.0) << 9)
    }

    /// Shifts every square one step diagonally toward a8, dropping squares that would leave the board.
    #[inline(always)]
    pub const fn north_west(self) -> Bitboard {
        Bitboard((self.0 & !Self::FILE_A.0) << 7)
    }

    /// Shifts every square one step diagonally toward h1, dropping squares that would leave the board.
    #[inline(always)]
    pub const fn south_east(self) -> Bitboard {
        Bitboard((self.0 & !Self::FILE_H.0) >> 7)
    }

    /// Shifts every square one step diagonally toward a1, dropping squares that would leave the board.
    #[inline(always)]
    pub const fn south_west(self) -> Bitboard {
        Bitboard((self.0 & !Self::FILE_A.0) >> 9)
    }
}

/// Iterator over the squares of a [`Bitboard`], lowest square first.
///
/// Created by [`Bitboard::into_iter`](IntoIterator::into_iter). It reports an
/// exact `size_hint`, so `collect` allocates once.
pub struct BitIter(u64);

impl Iterator for BitIter {
    type Item = Square;

    #[inline(always)]
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            let sq = Square(self.0.trailing_zeros() as u8);
            self.0 &= self.0 - 1;
            Some(sq)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.count_ones() as usize;
        (n, Some(n))
    }
}

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = BitIter;

    #[inline(always)]
    fn into_iter(self) -> BitIter {
        BitIter(self.0)
    }
}

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:tt) => {
        impl $trait for Bitboard {
            type Output = Bitboard;

            #[inline(always)]
            fn $method(self, rhs: Bitboard) -> Bitboard {
                Bitboard(self.0 $op rhs.0)
            }
        }

        impl $assign_trait for Bitboard {
            #[inline(always)]
            fn $assign_method(&mut self, rhs: Bitboard) {
                self.0 = self.0 $op rhs.0;
            }
        }
    };
}

impl_binop!(BitAnd, bitand, BitAndAssign, bitand_assign, &);
impl_binop!(BitOr, bitor, BitOrAssign, bitor_assign, |);
impl_binop!(BitXor, bitxor, BitXorAssign, bitxor_assign, ^);

impl Not for Bitboard {
    type Output = Bitboard;

    #[inline(always)]
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = Square::new(file, rank);
                write!(f, "{} ", if self.contains(sq) { 'X' } else { '.' })?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
