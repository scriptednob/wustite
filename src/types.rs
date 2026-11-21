//! Core board types shared across the crate: [`Color`], [`PieceType`],
//! [`Piece`], [`Square`], and the 16-bit [`Move`].
//!
//! Everything here is `Copy`, small, and backed by plain integers so indexing
//! arrays has zero overhead. Most accessors are `const fn` so lookup tables can
//! be built at compile time.
//!
//! # Index layout
//!
//! | Type | Range | Convention |
//! | --- | --- | --- |
//! | [`Color`] | `0..2` | `White = 0`, `Black = 1` |
//! | [`PieceType`] | `0..6` | pawn, knight, bishop, rook, queen, king |
//! | [`Piece`] | `0..12` | white pieces (`0..6`), then black pieces (`6..12`) |
//! | [`Square`] | `0..64` | `rank * 8 + file` (`a1 = 0`, `h1 = 7`, `a8 = 56`, `h8 = 63`) |
//!
//! # Move bit layout
//!
//! A [`Move`] fits in one `u16`: bits `0..6` hold the origin square, bits
//! `6..12` hold the destination square, and bits `12..16` hold the [`flag`]
//! bits (bit 2 = capture, bit 3 = promotion). All zeros (`a1a1`, quiet) is
//! reserved for [`Move::NONE`].
//!
//! ```
//! use wustite::{Color, Move, Piece, PieceType, Square};
//! use wustite::types::flag;
//!
//! let mv = Move::new(Square::E2, Square::E4, flag::DOUBLE_PUSH);
//! assert_eq!(mv.to_string(), "e2e4");
//! assert!(mv.is_double_push());
//! assert!(!mv.is_capture());
//!
//! let white_knight = Piece::new(Color::White, PieceType::Knight);
//! assert_eq!(white_knight.to_char(), 'N');
//! assert_eq!(white_knight.color(), Color::White);
//! ```

use core::fmt;

use crate::bitboard::Bitboard;

/// Side to move or piece color.
///
/// Use [`Color::idx`] to index 2-element arrays and `!color` (or
/// [`Color::flip`]) to get the opponent.
///
/// ```
/// use wustite::Color;
///
/// assert_eq!(!Color::White, Color::Black);
/// assert_eq!(Color::Black.idx(), 1);
/// assert_eq!(Color::ALL, [Color::White, Color::Black]);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Color {
    /// White (`0`).
    White = 0,
    /// Black (`1`).
    Black = 1,
}

impl Color {
    /// Both colors in index order (`[White, Black]`).
    pub const ALL: [Color; 2] = [Color::White, Color::Black];

    /// Array index (`0` for white, `1` for black).
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }

    /// Return the opposite side (same as `!self`).
    #[inline(always)]
    pub const fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

impl core::ops::Not for Color {
    type Output = Color;

    #[inline(always)]
    fn not(self) -> Color {
        self.flip()
    }
}

/// Piece kind without color (`0..6`, pawn through king).
///
/// Bitboard arrays in [`crate::board::Position`], value tables in
/// [`crate::eval`] and [`crate::see`], and NNUE feature offsets in
/// [`crate::nnue`] all rely on this exact ordering.
///
/// ```
/// use wustite::PieceType;
///
/// assert_eq!(PieceType::Pawn.idx(), 0);
/// assert_eq!(PieceType::King.idx(), 5);
/// assert_eq!(PieceType::from_idx(4), PieceType::Queen);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[repr(u8)]
pub enum PieceType {
    /// Pawn (`0`).
    Pawn = 0,
    /// Knight (`1`).
    Knight = 1,
    /// Bishop (`2`).
    Bishop = 2,
    /// Rook (`3`).
    Rook = 3,
    /// Queen (`4`).
    Queen = 4,
    /// King (`5`).
    King = 5,
}

impl PieceType {
    /// All six piece types in index order.
    pub const ALL: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    /// Array index in `0..6`.
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }

    /// Convert an index in `0..6` back to a [`PieceType`].
    ///
    /// Panics if `i >= 6`.
    #[inline(always)]
    pub const fn from_idx(i: usize) -> PieceType {
        Self::ALL[i]
    }
}

/// Colored piece (`0..12`): all white pieces (`0..6`), then all black pieces (`6..12`).
///
/// Index is `color * 6 + piece_type`. Stored directly in the mailbox of
/// [`crate::board::Position`].
///
/// ```
/// use wustite::{Color, Piece, PieceType};
///
/// let piece = Piece::new(Color::Black, PieceType::Queen);
/// assert_eq!(piece, Piece::BlackQueen);
/// assert_eq!(piece.idx(), 10);
/// assert_eq!(piece.piece_type(), PieceType::Queen);
/// assert_eq!(Piece::from_char('q'), Some(Piece::BlackQueen));
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Piece {
    /// White pawn (`0`).
    WhitePawn = 0,
    /// White knight (`1`).
    WhiteKnight,
    /// White bishop (`2`).
    WhiteBishop,
    /// White rook (`3`).
    WhiteRook,
    /// White queen (`4`).
    WhiteQueen,
    /// White king (`5`).
    WhiteKing,
    /// Black pawn (`6`).
    BlackPawn,
    /// Black knight (`7`).
    BlackKnight,
    /// Black bishop (`8`).
    BlackBishop,
    /// Black rook (`9`).
    BlackRook,
    /// Black queen (`10`).
    BlackQueen,
    /// Black king (`11`).
    BlackKing,
}

impl Piece {
    /// All twelve colored pieces in index order.
    pub const ALL: [Piece; 12] = [
        Piece::WhitePawn,
        Piece::WhiteKnight,
        Piece::WhiteBishop,
        Piece::WhiteRook,
        Piece::WhiteQueen,
        Piece::WhiteKing,
        Piece::BlackPawn,
        Piece::BlackKnight,
        Piece::BlackBishop,
        Piece::BlackRook,
        Piece::BlackQueen,
        Piece::BlackKing,
    ];

    const CHARS: [char; 12] = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'];

    /// Combine a color and piece type into a [`Piece`].
    ///
    /// ```
    /// use wustite::{Color, Piece, PieceType};
    ///
    /// assert_eq!(Piece::new(Color::White, PieceType::Rook), Piece::WhiteRook);
    /// ```
    #[inline(always)]
    pub const fn new(color: Color, piece_type: PieceType) -> Piece {
        Self::ALL[color.idx() * 6 + piece_type.idx()]
    }

    /// Array index in `0..12`.
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }

    /// Side this piece belongs to.
    #[inline(always)]
    pub const fn color(self) -> Color {
        if (self as u8) < 6 {
            Color::White
        } else {
            Color::Black
        }
    }

    /// Piece type without color.
    #[inline(always)]
    pub const fn piece_type(self) -> PieceType {
        PieceType::ALL[self.idx() % 6]
    }

    /// FEN character (`PNBRQK` for white, `pnbrqk` for black).
    pub fn to_char(self) -> char {
        Self::CHARS[self.idx()]
    }

    /// Parse a FEN piece letter, or return `None` for anything else.
    ///
    /// ```
    /// use wustite::Piece;
    ///
    /// assert_eq!(Piece::from_char('K'), Some(Piece::WhiteKing));
    /// assert_eq!(Piece::from_char('x'), None);
    /// ```
    pub fn from_char(c: char) -> Option<Piece> {
        Self::CHARS
            .iter()
            .position(|&x| x == c)
            .map(|i| Self::ALL[i])
    }
}

/// Board square in `0..64`, laid out as `rank * 8 + file` (`a1 = 0`, `h8 = 63`).
///
/// With this layout, stepping north is `+8`, stepping east is `+1`, and
/// flipping a square vertically for black's perspective is just `idx ^ 56`.
/// Named constants `Square::A1` through `Square::H8` are provided below.
///
/// ```
/// use wustite::Square;
///
/// let e4 = Square::new(4, 3);
/// assert_eq!(e4, Square::E4);
/// assert_eq!(e4.file(), 4);
/// assert_eq!(e4.rank(), 3);
/// assert_eq!(e4.to_string(), "e4");
/// assert_eq!(Square::parse("e4"), Some(Square::E4));
/// assert_eq!(Square::parse("i9"), None);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Square(pub u8);

impl Square {
    /// Total squares on a chessboard.
    pub const COUNT: usize = 64;

    /// Build a square from zero-based `file` (`0..8`) and `rank` (`0..8`). Caller ensures both are in range.
    #[inline(always)]
    pub const fn new(file: u8, rank: u8) -> Square {
        Square(rank * 8 + file)
    }

    /// Wrap a raw square index (`0..64`) without range checking.
    #[inline(always)]
    pub const fn from_idx(i: usize) -> Square {
        Square(i as u8)
    }

    /// Array index in `0..64`.
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }

    /// Zero-based file (`0` = a-file, `7` = h-file).
    #[inline(always)]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    /// Zero-based rank (`0` = rank 1, `7` = rank 8).
    #[inline(always)]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    /// Single-bit [`Bitboard`] with only this square set.
    ///
    /// ```
    /// use wustite::Square;
    ///
    /// assert_eq!(Square::A1.bb().0, 1);
    /// assert_eq!(Square::H8.bb().0, 1 << 63);
    /// ```
    #[inline(always)]
    pub const fn bb(self) -> Bitboard {
        Bitboard(1u64 << self.0)
    }

    /// Shift this square by `delta` indices (e.g. `+8` for one rank up). Caller guarantees the target stays on the board.
    #[inline(always)]
    pub const fn offset(self, delta: i8) -> Square {
        Square((self.0 as i8 + delta) as u8)
    }

    /// Parse algebraic square notation like `"e4"`, or return `None` if malformed.
    pub fn parse(text: &str) -> Option<Square> {
        let bytes = text.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let file = bytes[0].wrapping_sub(b'a');
        let rank = bytes[1].wrapping_sub(b'1');
        if file < 8 && rank < 8 {
            Some(Square::new(file, rank))
        } else {
            None
        }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            (b'a' + self.file()) as char,
            (b'1' + self.rank()) as char
        )
    }
}

impl Square {
    /// Square `a1`.
    pub const A1: Square = Square(0);
    /// Square `b1`.
    pub const B1: Square = Square(1);
    /// Square `c1`.
    pub const C1: Square = Square(2);
    /// Square `d1`.
    pub const D1: Square = Square(3);
    /// Square `e1`.
    pub const E1: Square = Square(4);
    /// Square `f1`.
    pub const F1: Square = Square(5);
    /// Square `g1`.
    pub const G1: Square = Square(6);
    /// Square `h1`.
    pub const H1: Square = Square(7);
    /// Square `a2`.
    pub const A2: Square = Square(8);
    /// Square `b2`.
    pub const B2: Square = Square(9);
    /// Square `c2`.
    pub const C2: Square = Square(10);
    /// Square `d2`.
    pub const D2: Square = Square(11);
    /// Square `e2`.
    pub const E2: Square = Square(12);
    /// Square `f2`.
    pub const F2: Square = Square(13);
    /// Square `g2`.
    pub const G2: Square = Square(14);
    /// Square `h2`.
    pub const H2: Square = Square(15);
    /// Square `a3`.
    pub const A3: Square = Square(16);
    /// Square `b3`.
    pub const B3: Square = Square(17);
    /// Square `c3`.
    pub const C3: Square = Square(18);
    /// Square `d3`.
    pub const D3: Square = Square(19);
    /// Square `e3`.
    pub const E3: Square = Square(20);
    /// Square `f3`.
    pub const F3: Square = Square(21);
    /// Square `g3`.
    pub const G3: Square = Square(22);
    /// Square `h3`.
    pub const H3: Square = Square(23);
    /// Square `a4`.
    pub const A4: Square = Square(24);
    /// Square `b4`.
    pub const B4: Square = Square(25);
    /// Square `c4`.
    pub const C4: Square = Square(26);
    /// Square `d4`.
    pub const D4: Square = Square(27);
    /// Square `e4`.
    pub const E4: Square = Square(28);
    /// Square `f4`.
    pub const F4: Square = Square(29);
    /// Square `g4`.
    pub const G4: Square = Square(30);
    /// Square `h4`.
    pub const H4: Square = Square(31);
    /// Square `a5`.
    pub const A5: Square = Square(32);
    /// Square `b5`.
    pub const B5: Square = Square(33);
    /// Square `c5`.
    pub const C5: Square = Square(34);
    /// Square `d5`.
    pub const D5: Square = Square(35);
    /// Square `e5`.
    pub const E5: Square = Square(36);
    /// Square `f5`.
    pub const F5: Square = Square(37);
    /// Square `g5`.
    pub const G5: Square = Square(38);
    /// Square `h5`.
    pub const H5: Square = Square(39);
    /// Square `a6`.
    pub const A6: Square = Square(40);
    /// Square `b6`.
    pub const B6: Square = Square(41);
    /// Square `c6`.
    pub const C6: Square = Square(42);
    /// Square `d6`.
    pub const D6: Square = Square(43);
    /// Square `e6`.
    pub const E6: Square = Square(44);
    /// Square `f6`.
    pub const F6: Square = Square(45);
    /// Square `g6`.
    pub const G6: Square = Square(46);
    /// Square `h6`.
    pub const H6: Square = Square(47);
    /// Square `a7`.
    pub const A7: Square = Square(48);
    /// Square `b7`.
    pub const B7: Square = Square(49);
    /// Square `c7`.
    pub const C7: Square = Square(50);
    /// Square `d7`.
    pub const D7: Square = Square(51);
    /// Square `e7`.
    pub const E7: Square = Square(52);
    /// Square `f7`.
    pub const F7: Square = Square(53);
    /// Square `g7`.
    pub const G7: Square = Square(54);
    /// Square `h7`.
    pub const H7: Square = Square(55);
    /// Square `a8`.
    pub const A8: Square = Square(56);
    /// Square `b8`.
    pub const B8: Square = Square(57);
    /// Square `c8`.
    pub const C8: Square = Square(58);
    /// Square `d8`.
    pub const D8: Square = Square(59);
    /// Square `e8`.
    pub const E8: Square = Square(60);
    /// Square `f8`.
    pub const F8: Square = Square(61);
    /// Square `g8`.
    pub const G8: Square = Square(62);
    /// Square `h8`.
    pub const H8: Square = Square(63);
}

/// 4-bit move flags stored in bits `12..16` of a [`Move`].
///
/// | Value | Constant | Meaning |
/// | --- | --- | --- |
/// | 0 | [`flag::QUIET`] | Normal non-capture |
/// | 1 | [`flag::DOUBLE_PUSH`] | Two-square pawn push |
/// | 2 | [`flag::KING_CASTLE`] | Castling kingside |
/// | 3 | [`flag::QUEEN_CASTLE`] | Castling queenside |
/// | 4 | [`flag::CAPTURE`] | Regular capture |
/// | 5 | [`flag::EN_PASSANT`] | En passant capture |
/// | 8..11 | `PROMO_*` | Quiet promotion (knight, bishop, rook, queen) |
/// | 12..15 | `PROMO_* \| PROMO_CAPTURE_BIT` | Capturing promotion |
///
/// Bit 2 (`4`) marks captures and bit 3 (`8`) marks promotions, so both checks
/// are single bit tests. The bottom two bits of any promotion pick the piece.
pub mod flag {
    /// Plain non-capturing move.
    pub const QUIET: u16 = 0;
    /// Two-square pawn advance.
    pub const DOUBLE_PUSH: u16 = 1;
    /// Kingside castling (encoded as the king moving two squares toward h).
    pub const KING_CASTLE: u16 = 2;
    /// Queenside castling (encoded as the king moving two squares toward a).
    pub const QUEEN_CASTLE: u16 = 3;
    /// Normal capture on the destination square.
    pub const CAPTURE: u16 = 4;
    /// En passant capture (captured pawn sits behind `to_sq`).
    pub const EN_PASSANT: u16 = 5;
    /// Quiet promotion to knight.
    pub const PROMO_KNIGHT: u16 = 8;
    /// Quiet promotion to bishop.
    pub const PROMO_BISHOP: u16 = 9;
    /// Quiet promotion to rook.
    pub const PROMO_ROOK: u16 = 10;
    /// Quiet promotion to queen.
    pub const PROMO_QUEEN: u16 = 11;
    /// Bit OR'd onto a `PROMO_*` flag when the promotion is also a capture.
    pub const PROMO_CAPTURE_BIT: u16 = 4;
}

/// Chess move packed into a `u16` (`from`, `to`, and 4-bit [`flag`]).
///
/// Doesn't store which piece moved—that's read from the board when playing the
/// move. Formats via `Display` as UCI long algebraic (`e2e4`, `e7e8q`, `0000`).
///
/// ```
/// use wustite::Move;
/// use wustite::Square;
/// use wustite::types::flag;
///
/// let promotion = Move::new(Square::E7, Square::E8, flag::PROMO_QUEEN);
/// assert_eq!(promotion.to_string(), "e7e8q");
/// assert!(promotion.is_promotion());
/// assert_eq!(promotion.promotion_piece(), Some(wustite::PieceType::Queen));
///
/// assert!(Move::NONE.is_none());
/// assert_eq!(Move::NONE.to_string(), "0000");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Move(u16);

impl Move {
    /// Empty/missing move sentinel (`0`), printed as `0000`.
    pub const NONE: Move = Move(0);

    /// Pack `from`, `to`, and [`flag`] bits into a `Move`.
    #[inline(always)]
    pub const fn new(from: Square, to: Square, flags: u16) -> Move {
        Move(from.0 as u16 | ((to.0 as u16) << 6) | (flags << 12))
    }

    /// Origin square.
    #[inline(always)]
    pub const fn from_sq(self) -> Square {
        Square((self.0 & 63) as u8)
    }

    /// Destination square (the king's target square for castling).
    #[inline(always)]
    pub const fn to_sq(self) -> Square {
        Square(((self.0 >> 6) & 63) as u8)
    }

    /// Top four flag bits (`0..16`); compare against constants in [`flag`].
    #[inline(always)]
    pub const fn flags(self) -> u16 {
        self.0 >> 12
    }

    /// Raw `u16` value, used when packing into the transposition table.
    #[inline(always)]
    pub const fn raw(self) -> u16 {
        self.0
    }

    /// Wrap a raw `u16` back into a `Move` without checking legality.
    #[inline(always)]
    pub const fn from_raw(raw: u16) -> Move {
        Move(raw)
    }

    /// `true` if this is [`Move::NONE`].
    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }

    /// `true` for normal captures, en passant, and capturing promotions.
    #[inline(always)]
    pub const fn is_capture(self) -> bool {
        self.flags() & 4 != 0
    }

    /// `true` for any pawn promotion (quiet or capturing).
    #[inline(always)]
    pub const fn is_promotion(self) -> bool {
        self.flags() & 8 != 0
    }

    /// `true` for an en passant capture.
    #[inline(always)]
    pub const fn is_en_passant(self) -> bool {
        self.flags() == flag::EN_PASSANT
    }

    /// `true` for kingside or queenside castling.
    #[inline(always)]
    pub const fn is_castle(self) -> bool {
        self.flags() == flag::KING_CASTLE || self.flags() == flag::QUEEN_CASTLE
    }

    /// `true` for a two-square pawn advance.
    #[inline(always)]
    pub const fn is_double_push(self) -> bool {
        self.flags() == flag::DOUBLE_PUSH
    }

    /// Promoted piece type, or `None` if this move isn't a promotion.
    #[inline(always)]
    pub const fn promotion_piece(self) -> Option<PieceType> {
        if self.is_promotion() {
            Some(match self.flags() & 3 {
                0 => PieceType::Knight,
                1 => PieceType::Bishop,
                2 => PieceType::Rook,
                _ => PieceType::Queen,
            })
        } else {
            None
        }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            return write!(f, "0000");
        }
        write!(f, "{}{}", self.from_sq(), self.to_sq())?;
        match self.promotion_piece() {
            Some(PieceType::Knight) => write!(f, "n"),
            Some(PieceType::Bishop) => write!(f, "b"),
            Some(PieceType::Rook) => write!(f, "r"),
            Some(PieceType::Queen) => write!(f, "q"),
            _ => Ok(()),
        }
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}
