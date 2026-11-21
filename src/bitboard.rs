//! 64-bit square sets (`Bitboard`), board shifts, and bit iteration.
//!
//! Bit `n` is set whenever `Square(n)` is in the set (`a1` = bit 0, `h8` = bit
//! 63). Bitwise `&`, `|`, `^`, and `!` handle set intersection, union,
//! symmetric difference, and complement in single instructions.
//!
//! Iterating over a bitboard pulls the lowest set bit with `trailing_zeros` and
//! clears it (`x &= x - 1`), so a loop only pays for the bits that are actually
//! set. Directional shifts (`east`, `west`, diagonals) mask off wraparound
//! files so bulk pawn moves take a single shift and mask.
//!
//! ```
//! use wustite::{Bitboard, Square};
//!
//! let set = Square::A1.bb() | Square::C3.bb() | Square::H8.bb();
//! let squares: Vec<Square> = set.into_iter().collect();
//! assert_eq!(squares, [Square::A1, Square::C3, Square::H8]);
//! assert_eq!(set.count(), 3);
//! ```

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use crate::types::Square;

/// Set of squares stored in a `u64`.
///
/// `Debug` prints an 8x8 board (rank 8 at the top), which makes debugging move
/// generation and pin masks much easier.
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
    /// Empty square set.
    pub const EMPTY: Bitboard = Bitboard(0);
    /// All 64 squares.
    pub const FULL: Bitboard = Bitboard(!0);

    /// All squares on the a-file.
    pub const FILE_A: Bitboard = Bitboard(0x0101_0101_0101_0101);
    /// All squares on the h-file.
    pub const FILE_H: Bitboard = Bitboard(0x8080_8080_8080_8080);

    /// Rank 1.
    pub const RANK_1: Bitboard = Bitboard(0x0000_0000_0000_00FF);
    /// Rank 2.
    pub const RANK_2: Bitboard = Bitboard(0x0000_0000_0000_FF00);
    /// Rank 3 (intermediate square for white double pawn pushes).
    pub const RANK_3: Bitboard = Bitboard(0x0000_0000_00FF_0000);
    /// Rank 6 (intermediate square for black double pawn pushes).
    pub const RANK_6: Bitboard = Bitboard(0x0000_FF00_0000_0000);
    /// Rank 7.
    pub const RANK_7: Bitboard = Bitboard(0x00FF_0000_0000_0000);
    /// Rank 8.
    pub const RANK_8: Bitboard = Bitboard(0xFF00_0000_0000_0000);

    /// `true` when no bits are set.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// `true` when at least one bit is set.
    #[inline(always)]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    /// Number of set squares (popcount).
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// `true` if two or more bits are set, using `x & (x - 1) != 0` instead of a full popcount.
    #[inline(always)]
    pub const fn more_than_one(self) -> bool {
        self.0 & self.0.wrapping_sub(1) != 0
    }

    /// `true` if `sq` is set.
    #[inline(always)]
    pub const fn contains(self, sq: Square) -> bool {
        (self.0 >> sq.0) & 1 != 0
    }

    /// Lowest set square. Only meaningful on non-empty bitboards (empty returns index 64).
    #[inline(always)]
    pub const fn lsb(self) -> Square {
        Square(self.0.trailing_zeros() as u8)
    }

    /// Clear and return the lowest set square. Caller guarantees the bitboard isn't empty.
    #[inline(always)]
    pub fn pop_lsb(&mut self) -> Square {
        let sq = self.lsb();
        self.0 &= self.0.wrapping_sub(1);
        sq
    }

    /// Shift every square one rank toward rank 8.
    #[inline(always)]
    pub const fn north(self) -> Bitboard {
        Bitboard(self.0 << 8)
    }

    /// Shift every square one rank toward rank 1.
    #[inline(always)]
    pub const fn south(self) -> Bitboard {
        Bitboard(self.0 >> 8)
    }

    /// Shift one file toward h, dropping the h-file so bits don't wrap.
    #[inline(always)]
    pub const fn east(self) -> Bitboard {
        Bitboard((self.0 & !Self::FILE_H.0) << 1)
    }

    /// Shift one file toward a, dropping the a-file so bits don't wrap.
    #[inline(always)]
    pub const fn west(self) -> Bitboard {
        Bitboard((self.0 & !Self::FILE_A.0) >> 1)
    }

    /// Shift diagonally toward h8 without wrapping off the h-file.
    #[inline(always)]
    pub const fn north_east(self) -> Bitboard {
        Bitboard((self.0 & !Self::FILE_H.0) << 9)
    }

    /// Shift diagonally toward a8 without wrapping off the a-file.
    #[inline(always)]
    pub const fn north_west(self) -> Bitboard {
        Bitboard((self.0 & !Self::FILE_A.0) << 7)
    }

    /// Shift diagonally toward h1 without wrapping off the h-file.
    #[inline(always)]
    pub const fn south_east(self) -> Bitboard {
        Bitboard((self.0 & !Self::FILE_H.0) >> 7)
    }

    /// Shift diagonally toward a1 without wrapping off the a-file.
    #[inline(always)]
    pub const fn south_west(self) -> Bitboard {
        Bitboard((self.0 & !Self::FILE_A.0) >> 9)
    }
}

/// Iterator yielding set squares from lowest bit to highest, with an exact `size_hint`.
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
