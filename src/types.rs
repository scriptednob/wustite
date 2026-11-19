//! Core vocabulary shared by every other module: colours, piece kinds, pieces,
//! squares and the packed 16-bit [`Move`].
//!
//! Everything here is `Copy`, small, and cheap to pass by value. The types are
//! thin wrappers over integers so that they can be used as array indices without
//! any conversion cost, and most accessors are `const fn` so that lookup tables
//! can be built at compile time.
//!
//! # Index conventions
//!
//! | Type | Range | Layout |
//! | --- | --- | --- |
//! | [`Color`] | `0..2` | `White = 0`, `Black = 1` |
//! | [`PieceType`] | `0..6` | pawn, knight, bishop, rook, queen, king |
//! | [`Piece`] | `0..12` | all six white pieces, then all six black pieces |
//! | [`Square`] | `0..64` | `index = rank * 8 + file`, so a1 = 0, h1 = 7, a8 = 56, h8 = 63 |
//!
//! # Move encoding
//!
//! A [`Move`] is a single `u16`:
//!
//! ```text
//!  15 14 13 12 | 11 10 9 8 7 6 | 5 4 3 2 1 0
//!    flags     |   to square   | from square
//! ```
//!
//! The four flag bits follow the common "from-to plus 4-bit flags" scheme. Bit 2
//! (value 4) means *capture* and bit 3 (value 8) means *promotion*, which makes
//! the most frequent questions ("is this a capture?", "is this a promotion?")
//! a single mask test. See the [`flag`] module for the full table.
//!
//! The all-zero move (a1 to a1, quiet) is reserved as [`Move::NONE`]. No legal
//! move can have identical origin and destination squares, so the encoding is
//! unambiguous.
//!
//! # Examples
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

/// The two sides of a chess game.
///
/// `Color` converts to an array index with [`Color::idx`], and `!color` (or
/// [`Color::flip`]) gives the opponent.
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
    /// The side that moves first. Index `0`.
    White = 0,
    /// The side that moves second. Index `1`.
    Black = 1,
}

impl Color {
    /// Both colours in index order, for `for color in Color::ALL` loops.
    pub const ALL: [Color; 2] = [Color::White, Color::Black];

    /// The colour as an array index: `0` for white, `1` for black.
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }

    /// The opposite colour. Also available through the `!` operator.
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

/// The six kinds of chess piece, ignoring colour.
///
/// The order (pawn, knight, bishop, rook, queen, king) is relied on throughout
/// the engine: per-type bitboards in [`crate::board::Position`], the piece-value
/// tables in [`crate::see`] and [`crate::eval`], and the NNUE feature layout in
/// [`crate::nnue`] are all indexed by [`PieceType::idx`].
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
    /// Pawn. Index `0`.
    Pawn = 0,
    /// Knight. Index `1`.
    Knight = 1,
    /// Bishop. Index `2`.
    Bishop = 2,
    /// Rook. Index `3`.
    Rook = 3,
    /// Queen. Index `4`.
    Queen = 4,
    /// King. Index `5`.
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

    /// The piece type as an array index in `0..6`.
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }

    /// The piece type with index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= 6`.
    #[inline(always)]
    pub const fn from_idx(i: usize) -> PieceType {
        Self::ALL[i]
    }
}

/// A coloured piece: one of twelve values.
///
/// The layout is "all white pieces, then all black pieces", each group in
/// [`PieceType`] order, so `index = color * 6 + piece_type`. [`Piece::new`],
/// [`Piece::color`] and [`Piece::piece_type`] convert between the combined and
/// split representations.
///
/// The mailbox inside [`crate::board::Position`] stores `Option<Piece>` per
/// square, and the NNUE feature index is derived from the piece's colour relative
/// to the viewing side.
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
    /// White pawn. Index `0`.
    WhitePawn = 0,
    /// White knight. Index `1`.
    WhiteKnight,
    /// White bishop. Index `2`.
    WhiteBishop,
    /// White rook. Index `3`.
    WhiteRook,
    /// White queen. Index `4`.
    WhiteQueen,
    /// White king. Index `5`.
    WhiteKing,
    /// Black pawn. Index `6`.
    BlackPawn,
    /// Black knight. Index `7`.
    BlackKnight,
    /// Black bishop. Index `8`.
    BlackBishop,
    /// Black rook. Index `9`.
    BlackRook,
    /// Black queen. Index `10`.
    BlackQueen,
    /// Black king. Index `11`.
    BlackKing,
}

impl Piece {
    /// All twelve pieces in index order.
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

    /// Combines a colour and a piece type into a [`Piece`].
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

    /// The piece as an array index in `0..12`.
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }

    /// The side the piece belongs to.
    #[inline(always)]
    pub const fn color(self) -> Color {
        if (self as u8) < 6 {
            Color::White
        } else {
            Color::Black
        }
    }

    /// The kind of piece, with colour discarded.
    #[inline(always)]
    pub const fn piece_type(self) -> PieceType {
        PieceType::ALL[self.idx() % 6]
    }

    /// The FEN letter: uppercase for white (`PNBRQK`), lowercase for black (`pnbrqk`).
    pub fn to_char(self) -> char {
        Self::CHARS[self.idx()]
    }

    /// Parses a FEN piece letter, returning `None` for any other character.
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

/// A board square, stored as an index in `0..64` with `index = rank * 8 + file`.
///
/// Square `0` is a1, square `7` is h1, square `56` is a8 and square `63` is h8.
/// Files and ranks are both zero-based, so file `0` is the a-file and rank `0` is
/// the first rank. This "little-endian rank-file" mapping means that moving one
/// rank up is `+8`, one file right is `+1`, and flipping a square vertically is
/// `index ^ 56`, which both [`crate::eval`] and [`crate::nnue`] use to view the
/// board from black's side.
///
/// All 64 squares are available as associated constants, `Square::A1` through
/// `Square::H8`.
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
    /// The number of squares on the board.
    pub const COUNT: usize = 64;

    /// The square at the given zero-based file and rank.
    ///
    /// The arguments are not range-checked; `file` and `rank` must both be below 8.
    #[inline(always)]
    pub const fn new(file: u8, rank: u8) -> Square {
        Square(rank * 8 + file)
    }

    /// The square with index `i`. The index must be below 64.
    #[inline(always)]
    pub const fn from_idx(i: usize) -> Square {
        Square(i as u8)
    }

    /// The square as an array index in `0..64`.
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }

    /// The zero-based file: `0` for the a-file up to `7` for the h-file.
    #[inline(always)]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    /// The zero-based rank: `0` for the first rank up to `7` for the eighth.
    #[inline(always)]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    /// A [`Bitboard`] with only this square set.
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

    /// The square `delta` indices away, for example `+8` for one rank up.
    ///
    /// The result is not range-checked. Callers use it only where the destination is
    /// known to be on the board, such as the square behind an en passant target.
    #[inline(always)]
    pub const fn offset(self, delta: i8) -> Square {
        Square((self.0 as i8 + delta) as u8)
    }

    /// Parses algebraic notation such as `"e4"`. Returns `None` unless the text is
    /// exactly one file letter `a`-`h` followed by one rank digit `1`-`8`.
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
    /// The square a1, index 0.
    pub const A1: Square = Square(0);
    /// The square b1, index 1.
    pub const B1: Square = Square(1);
    /// The square c1, index 2.
    pub const C1: Square = Square(2);
    /// The square d1, index 3.
    pub const D1: Square = Square(3);
    /// The square e1, index 4.
    pub const E1: Square = Square(4);
    /// The square f1, index 5.
    pub const F1: Square = Square(5);
    /// The square g1, index 6.
    pub const G1: Square = Square(6);
    /// The square h1, index 7.
    pub const H1: Square = Square(7);
    /// The square a2, index 8.
    pub const A2: Square = Square(8);
    /// The square b2, index 9.
    pub const B2: Square = Square(9);
    /// The square c2, index 10.
    pub const C2: Square = Square(10);
    /// The square d2, index 11.
    pub const D2: Square = Square(11);
    /// The square e2, index 12.
    pub const E2: Square = Square(12);
    /// The square f2, index 13.
    pub const F2: Square = Square(13);
    /// The square g2, index 14.
    pub const G2: Square = Square(14);
    /// The square h2, index 15.
    pub const H2: Square = Square(15);
    /// The square a3, index 16.
    pub const A3: Square = Square(16);
    /// The square b3, index 17.
    pub const B3: Square = Square(17);
    /// The square c3, index 18.
    pub const C3: Square = Square(18);
    /// The square d3, index 19.
    pub const D3: Square = Square(19);
    /// The square e3, index 20.
    pub const E3: Square = Square(20);
    /// The square f3, index 21.
    pub const F3: Square = Square(21);
    /// The square g3, index 22.
    pub const G3: Square = Square(22);
    /// The square h3, index 23.
    pub const H3: Square = Square(23);
    /// The square a4, index 24.
    pub const A4: Square = Square(24);
    /// The square b4, index 25.
    pub const B4: Square = Square(25);
    /// The square c4, index 26.
    pub const C4: Square = Square(26);
    /// The square d4, index 27.
    pub const D4: Square = Square(27);
    /// The square e4, index 28.
    pub const E4: Square = Square(28);
    /// The square f4, index 29.
    pub const F4: Square = Square(29);
    /// The square g4, index 30.
    pub const G4: Square = Square(30);
    /// The square h4, index 31.
    pub const H4: Square = Square(31);
    /// The square a5, index 32.
    pub const A5: Square = Square(32);
    /// The square b5, index 33.
    pub const B5: Square = Square(33);
    /// The square c5, index 34.
    pub const C5: Square = Square(34);
    /// The square d5, index 35.
    pub const D5: Square = Square(35);
    /// The square e5, index 36.
    pub const E5: Square = Square(36);
    /// The square f5, index 37.
    pub const F5: Square = Square(37);
    /// The square g5, index 38.
    pub const G5: Square = Square(38);
    /// The square h5, index 39.
    pub const H5: Square = Square(39);
    /// The square a6, index 40.
    pub const A6: Square = Square(40);
    /// The square b6, index 41.
    pub const B6: Square = Square(41);
    /// The square c6, index 42.
    pub const C6: Square = Square(42);
    /// The square d6, index 43.
    pub const D6: Square = Square(43);
    /// The square e6, index 44.
    pub const E6: Square = Square(44);
    /// The square f6, index 45.
    pub const F6: Square = Square(45);
    /// The square g6, index 46.
    pub const G6: Square = Square(46);
    /// The square h6, index 47.
    pub const H6: Square = Square(47);
    /// The square a7, index 48.
    pub const A7: Square = Square(48);
    /// The square b7, index 49.
    pub const B7: Square = Square(49);
    /// The square c7, index 50.
    pub const C7: Square = Square(50);
    /// The square d7, index 51.
    pub const D7: Square = Square(51);
    /// The square e7, index 52.
    pub const E7: Square = Square(52);
    /// The square f7, index 53.
    pub const F7: Square = Square(53);
    /// The square g7, index 54.
    pub const G7: Square = Square(54);
    /// The square h7, index 55.
    pub const H7: Square = Square(55);
    /// The square a8, index 56.
    pub const A8: Square = Square(56);
    /// The square b8, index 57.
    pub const B8: Square = Square(57);
    /// The square c8, index 58.
    pub const C8: Square = Square(58);
    /// The square d8, index 59.
    pub const D8: Square = Square(59);
    /// The square e8, index 60.
    pub const E8: Square = Square(60);
    /// The square f8, index 61.
    pub const F8: Square = Square(61);
    /// The square g8, index 62.
    pub const G8: Square = Square(62);
    /// The square h8, index 63.
    pub const H8: Square = Square(63);
}

/// Move flag values stored in the top four bits of a [`Move`].
///
/// | Value | Constant | Meaning |
/// | --- | --- | --- |
/// | 0 | [`flag::QUIET`] | Non-capturing, non-special move |
/// | 1 | [`flag::DOUBLE_PUSH`] | Pawn advancing two squares |
/// | 2 | [`flag::KING_CASTLE`] | Castling short (king to the g-file) |
/// | 3 | [`flag::QUEEN_CASTLE`] | Castling long (king to the c-file) |
/// | 4 | [`flag::CAPTURE`] | Ordinary capture |
/// | 5 | [`flag::EN_PASSANT`] | En passant capture |
/// | 8 | [`flag::PROMO_KNIGHT`] | Promotion to a knight |
/// | 9 | [`flag::PROMO_BISHOP`] | Promotion to a bishop |
/// | 10 | [`flag::PROMO_ROOK`] | Promotion to a rook |
/// | 11 | [`flag::PROMO_QUEEN`] | Promotion to a queen |
/// | 12 to 15 | promotion `\|` [`flag::PROMO_CAPTURE_BIT`] | Capturing promotion |
///
/// Bit 2 of the flags marks captures and bit 3 marks promotions, so
/// `flags & 4 != 0` is "is a capture" and `flags & 8 != 0` is "is a promotion".
/// The low two bits of a promotion flag select the piece
/// (`0` knight, `1` bishop, `2` rook, `3` queen).
pub mod flag {
    /// Plain non-capturing move.
    pub const QUIET: u16 = 0;
    /// Pawn double push. Sets the en passant square when an enemy pawn can use it.
    pub const DOUBLE_PUSH: u16 = 1;
    /// King-side castling. The move is encoded as the king moving two squares toward the h-file.
    pub const KING_CASTLE: u16 = 2;
    /// Queen-side castling. The move is encoded as the king moving two squares toward the a-file.
    pub const QUEEN_CASTLE: u16 = 3;
    /// Ordinary capture of the piece on the destination square.
    pub const CAPTURE: u16 = 4;
    /// En passant capture. The captured pawn is not on the destination square.
    pub const EN_PASSANT: u16 = 5;
    /// Quiet promotion to a knight.
    pub const PROMO_KNIGHT: u16 = 8;
    /// Quiet promotion to a bishop.
    pub const PROMO_BISHOP: u16 = 9;
    /// Quiet promotion to a rook.
    pub const PROMO_ROOK: u16 = 10;
    /// Quiet promotion to a queen.
    pub const PROMO_QUEEN: u16 = 11;
    /// OR this into a promotion flag to mark the promotion as a capture.
    pub const PROMO_CAPTURE_BIT: u16 = 4;
}

/// A chess move packed into 16 bits: origin square, destination square and flags.
///
/// See the [module documentation](self) for the bit layout. A `Move` does not
/// know which piece is moving; that is looked up on the board when the move is
/// made. The textual form (`Display`) is long algebraic notation as used by UCI,
/// for example `e2e4` or `e7e8q`.
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
    /// The null move, used as "no move" in tables and results. It prints as `0000`.
    pub const NONE: Move = Move(0);

    /// Packs an origin square, a destination square and a [`flag`] value into a move.
    #[inline(always)]
    pub const fn new(from: Square, to: Square, flags: u16) -> Move {
        Move(from.0 as u16 | ((to.0 as u16) << 6) | (flags << 12))
    }

    /// The square the piece moves from.
    #[inline(always)]
    pub const fn from_sq(self) -> Square {
        Square((self.0 & 63) as u8)
    }

    /// The square the piece moves to. For castling this is the king's destination.
    #[inline(always)]
    pub const fn to_sq(self) -> Square {
        Square(((self.0 >> 6) & 63) as u8)
    }

    /// The four flag bits as a value in `0..16`. Compare against the [`flag`] constants.
    #[inline(always)]
    pub const fn flags(self) -> u16 {
        self.0 >> 12
    }

    /// The raw 16-bit encoding, as stored in the transposition table.
    #[inline(always)]
    pub const fn raw(self) -> u16 {
        self.0
    }

    /// Rebuilds a move from [`Move::raw`]. Any `u16` is accepted; no legality is implied.
    #[inline(always)]
    pub const fn from_raw(raw: u16) -> Move {
        Move(raw)
    }

    /// Whether this is [`Move::NONE`].
    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }

    /// Whether the move captures a piece, including en passant and capturing promotions.
    #[inline(always)]
    pub const fn is_capture(self) -> bool {
        self.flags() & 4 != 0
    }

    /// Whether the move promotes a pawn.
    #[inline(always)]
    pub const fn is_promotion(self) -> bool {
        self.flags() & 8 != 0
    }

    /// Whether the move is an en passant capture.
    #[inline(always)]
    pub const fn is_en_passant(self) -> bool {
        self.flags() == flag::EN_PASSANT
    }

    /// Whether the move is a king-side or queen-side castle.
    #[inline(always)]
    pub const fn is_castle(self) -> bool {
        self.flags() == flag::KING_CASTLE || self.flags() == flag::QUEEN_CASTLE
    }

    /// Whether the move is a two-square pawn advance.
    #[inline(always)]
    pub const fn is_double_push(self) -> bool {
        self.flags() == flag::DOUBLE_PUSH
    }

    /// The piece a pawn promotes to, or `None` when the move is not a promotion.
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
