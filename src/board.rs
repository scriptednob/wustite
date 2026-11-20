//! The chess position and everything needed to move through the game tree.
//!
//! [`Position`] is the central data structure of the engine. It holds one
//! bitboard per piece type, one per colour, a 64-entry mailbox that answers
//! "which piece is on this square?", the side to move, castling rights, the en
//! passant target, the clocks, and an incrementally maintained Zobrist hash.
//!
//! # Copy-make
//!
//! Moves are applied with *copy-make*: [`Position::make_move`] takes `&self` and
//! returns a **new** position. There is no `unmake`. Instead the search keeps the
//! parent position on the stack and simply drops the child when it returns.
//!
//! Copy-make suits a multi-threaded engine with a cache-conscious layout. The struct is exactly 192 bytes (three 64-byte cache lines, enforced
//! by `repr(C, align(64))`) and `Copy`, so making a move is a handful of
//! cache-line copies followed by a few XORs. In exchange the search never has to
//! store undo information (captured piece, previous castling rights, previous en
//! passant square, previous hash, previous clock) or run an unmake routine, and
//! there is no way to corrupt state by forgetting to restore it.
//!
//! # What `make_move` updates
//!
//! * the piece and colour bitboards and the mailbox, for the moved piece, a
//!   captured piece (including the en passant victim, which is not on the
//!   destination square) and the rook when castling;
//! * the Zobrist hash, by XOR-ing out and in only the features that changed;
//! * castling rights, through a 64-entry mask: any move from or to a king or rook
//!   home square removes the matching rights, which also covers captures of an
//!   unmoved rook;
//! * the en passant square, which is set after a double push **only if an enemy
//!   pawn is able to capture there**. This keeps the hash identical for
//!   positions that differ only by an unusable en passant square, which improves
//!   transposition-table hits and repetition detection;
//! * the halfmove clock (reset by pawn moves and captures) and the fullmove number.
//!
//! `make_move` assumes the move is legal. Passing an illegal move is a logic error
//! (debug builds assert in the obvious case of an empty origin square).
//!
//! # Forsyth-Edwards Notation
//!
//! [`Position::from_fen`] and [`Position::to_fen`] convert to and from FEN.
//! Parsing validates the input: exactly one king per side, no pawns on the first
//! or last rank, at most eight pawns and sixteen pieces a side, the side not to
//! move not in check, and a well-formed en passant square. Castling rights that do
//! not match the pieces on their home squares are silently dropped, and an en
//! passant square that no pawn can capture on is ignored. The halfmove and
//! fullmove fields are optional.
//!
//! # Examples
//!
//! ```
//! use wustite::Position;
//! use wustite::movegen::{legal_moves, parse_move};
//!
//! let pos = Position::startpos();
//! assert_eq!(pos.to_fen(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//! assert_eq!(legal_moves(&pos).len(), 20);
//!
//! let e4 = parse_move(&pos, "e2e4").expect("e2e4 is legal");
//! let after = pos.make_move(e4);
//! assert_eq!(after.side_to_move(), wustite::Color::Black);
//! assert_eq!(pos.to_fen(), Position::startpos().to_fen());
//! assert_eq!(after.hash(), after.compute_hash());
//! ```

use core::fmt;
use core::str::FromStr;

use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::types::{Color, Move, Piece, PieceType, Square, flag};
use crate::zobrist::ZOBRIST;

/// Castling-rights bit: white may castle short (e1 and h1 unmoved).
pub const WHITE_KING_SIDE: u8 = 1;
/// Castling-rights bit: white may castle long (e1 and a1 unmoved).
pub const WHITE_QUEEN_SIDE: u8 = 2;
/// Castling-rights bit: black may castle short (e8 and h8 unmoved).
pub const BLACK_KING_SIDE: u8 = 4;
/// Castling-rights bit: black may castle long (e8 and a8 unmoved).
pub const BLACK_QUEEN_SIDE: u8 = 8;
/// All four castling-rights bits set. Castling rights are a 4-bit mask, so valid values are `0..=ALL_CASTLING`.
pub const ALL_CASTLING: u8 = 15;

/// The FEN string of the standard starting position.
pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Sentinel stored in the en passant field when no en passant capture is available.
const NO_SQUARE: u8 = 64;

/// Builds the castling-rights mask table. For each square, the value is the set
/// of rights that survive a move that starts or ends on it: the king's home
/// square clears both of a side's rights, each rook's home square clears one.
/// Every other square keeps all rights.
const fn build_castle_mask() -> [u8; 64] {
    let mut mask = [ALL_CASTLING; 64];
    mask[Square::A1.idx()] = ALL_CASTLING & !WHITE_QUEEN_SIDE;
    mask[Square::E1.idx()] = ALL_CASTLING & !(WHITE_KING_SIDE | WHITE_QUEEN_SIDE);
    mask[Square::H1.idx()] = ALL_CASTLING & !WHITE_KING_SIDE;
    mask[Square::A8.idx()] = ALL_CASTLING & !BLACK_QUEEN_SIDE;
    mask[Square::E8.idx()] = ALL_CASTLING & !(BLACK_KING_SIDE | BLACK_QUEEN_SIDE);
    mask[Square::H8.idx()] = ALL_CASTLING & !BLACK_KING_SIDE;
    mask
}

/// The castling-rights mask, indexed by square. See [`build_castle_mask`].
static CASTLE_MASK: [u8; 64] = build_castle_mask();

/// Why a FEN string could not be turned into a [`Position`].
///
/// The `Display` implementation produces a readable message; the engine prints it
/// in a UCI `info string` line when a GUI sends a bad `position fen`.
///
/// ```
/// use wustite::Position;
/// use wustite::board::FenError;
///
/// assert!(matches!(
///     Position::from_fen("8/8/8/8/8/8/8/8 w - - 0 1"),
///     Err(FenError::Invalid(_))
/// ));
/// assert!(matches!(Position::from_fen(""), Err(FenError::MissingField(_))));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FenError {
    /// A required field (piece placement or side to move) was absent. The payload names the field.
    MissingField(&'static str),
    /// A field was present but unacceptable. The payload describes the problem, for example `"each side needs exactly one king"`.
    Invalid(&'static str),
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FenError::MissingField(name) => write!(f, "FEN is missing the {name} field"),
            FenError::Invalid(reason) => write!(f, "invalid FEN: {reason}"),
        }
    }
}

impl std::error::Error for FenError {}

/// A complete chess position. See the [module documentation](crate::board) for the design.
///
/// The struct is `Copy`, 192 bytes and 64-byte aligned. Its fields are private so
/// that the redundant representations (piece bitboards, colour bitboards, mailbox
/// and hash) can never be put out of sync from outside; the accessors below give
/// read access and [`Position::make_move`] is the only way to advance a position.
///
/// Two positions compare equal when every field, including the clocks, is equal.
/// To compare positions for repetition purposes use [`Position::hash`].
///
/// ```
/// use wustite::{Color, PieceType, Position, Square};
///
/// let pos: Position = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1".parse().unwrap();
/// assert_eq!(pos.side_to_move(), Color::White);
/// assert_eq!(pos.king_square(Color::Black), Square::E8);
/// assert_eq!(pos.pieces(Color::White, PieceType::Pawn), Square::E2.bb());
/// assert!(!pos.in_check());
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C, align(64))]
pub struct Position {
    pieces: [Bitboard; 6],
    colors: [Bitboard; 2],
    mailbox: [Option<Piece>; 64],
    hash: u64,
    fullmove: u16,
    halfmove: u16,
    side: Color,
    castling: u8,
    ep: u8,
}

impl Position {
    /// A board with no pieces, white to move. Used as the starting point while parsing FEN.
    fn empty() -> Position {
        Position {
            pieces: [Bitboard::EMPTY; 6],
            colors: [Bitboard::EMPTY; 2],
            mailbox: [None; 64],
            hash: 0,
            fullmove: 1,
            halfmove: 0,
            side: Color::White,
            castling: 0,
            ep: NO_SQUARE,
        }
    }

    /// The standard starting position.
    pub fn startpos() -> Position {
        match Position::from_fen(START_FEN) {
            Ok(pos) => pos,
            Err(_) => Position::empty(),
        }
    }

    /// The side whose turn it is.
    #[inline(always)]
    pub fn side_to_move(&self) -> Color {
        self.side
    }

    /// The Zobrist hash of the position, maintained incrementally.
    ///
    /// It covers piece placement, castling rights, a usable en passant file and the
    /// side to move, but not the halfmove or fullmove counters, so it is suitable as
    /// a transposition-table key and for repetition detection.
    #[inline(always)]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    /// The castling rights as a bitmask of [`WHITE_KING_SIDE`], [`WHITE_QUEEN_SIDE`], [`BLACK_KING_SIDE`] and [`BLACK_QUEEN_SIDE`].
    #[inline(always)]
    pub fn castling(&self) -> u8 {
        self.castling
    }

    /// Plies since the last capture or pawn move, used for the fifty-move rule (a draw claim becomes available at 100).
    #[inline(always)]
    pub fn halfmove_clock(&self) -> u16 {
        self.halfmove
    }

    /// The move number as in FEN. It starts at 1 and increments after black moves.
    #[inline(always)]
    pub fn fullmove_number(&self) -> u16 {
        self.fullmove
    }

    /// The square a pawn may capture onto en passant, if any. It is only `Some` when an enemy pawn can really make the capture.
    #[inline(always)]
    pub fn ep_square(&self) -> Option<Square> {
        if self.ep == NO_SQUARE {
            None
        } else {
            Some(Square(self.ep))
        }
    }

    /// The piece standing on `sq`, if any. A single mailbox read.
    #[inline(always)]
    pub fn piece_on(&self, sq: Square) -> Option<Piece> {
        self.mailbox[sq.idx()]
    }

    /// The squares occupied by pieces of one colour and type, for example white rooks.
    #[inline(always)]
    pub fn pieces(&self, color: Color, piece_type: PieceType) -> Bitboard {
        self.pieces[piece_type.idx()] & self.colors[color.idx()]
    }

    /// The squares occupied by a piece type of either colour.
    #[inline(always)]
    pub fn piece_bb(&self, piece_type: PieceType) -> Bitboard {
        self.pieces[piece_type.idx()]
    }

    /// The squares occupied by all pieces of one colour.
    #[inline(always)]
    pub fn color_bb(&self, color: Color) -> Bitboard {
        self.colors[color.idx()]
    }

    /// All occupied squares.
    #[inline(always)]
    pub fn occupied(&self) -> Bitboard {
        self.colors[0] | self.colors[1]
    }

    /// The square of `color`'s king. Every valid position has exactly one king per side.
    #[inline(always)]
    pub fn king_square(&self, color: Color) -> Square {
        self.pieces(color, PieceType::King).lsb()
    }

    /// All pieces of **both** colours that attack `sq`, computed as if the board
    /// occupancy were `occ`.
    ///
    /// Passing a modified occupancy is the point of the parameter: static exchange
    /// evaluation removes pieces from `occ` as captures are played so that x-ray
    /// attackers behind them appear, and en passant legality checks remove the two
    /// pawns involved to see whether the king would be exposed.
    #[inline(always)]
    pub fn attackers_to(&self, sq: Square, occ: Bitboard) -> Bitboard {
        let pawns = (pawn_attacks(Color::White, sq) & self.pieces(Color::Black, PieceType::Pawn))
            | (pawn_attacks(Color::Black, sq) & self.pieces(Color::White, PieceType::Pawn));
        let knights = knight_attacks(sq) & self.pieces[PieceType::Knight.idx()];
        let kings = king_attacks(sq) & self.pieces[PieceType::King.idx()];
        let diagonal = self.pieces[PieceType::Bishop.idx()] | self.pieces[PieceType::Queen.idx()];
        let straight = self.pieces[PieceType::Rook.idx()] | self.pieces[PieceType::Queen.idx()];
        pawns
            | knights
            | kings
            | (bishop_attacks(sq, occ) & diagonal)
            | (rook_attacks(sq, occ) & straight)
    }

    /// Whether any piece of colour `by` attacks `sq` given the occupancy `occ`.
    ///
    /// This is a boolean query that stops at the first attacker found, cheaper than
    /// building the full [`Position::attackers_to`] set. King-move legality uses it
    /// with the king removed from `occ`, so a king cannot step backwards along the
    /// line of a checking slider.
    #[inline(always)]
    pub fn attacked_by(&self, sq: Square, by: Color, occ: Bitboard) -> bool {
        let theirs = self.colors[by.idx()];
        let diagonal =
            (self.pieces[PieceType::Bishop.idx()] | self.pieces[PieceType::Queen.idx()]) & theirs;
        let straight =
            (self.pieces[PieceType::Rook.idx()] | self.pieces[PieceType::Queen.idx()]) & theirs;
        (pawn_attacks(by.flip(), sq) & self.pieces[PieceType::Pawn.idx()] & theirs).any()
            || (knight_attacks(sq) & self.pieces[PieceType::Knight.idx()] & theirs).any()
            || (king_attacks(sq) & self.pieces[PieceType::King.idx()] & theirs).any()
            || (bishop_attacks(sq, occ) & diagonal).any()
            || (rook_attacks(sq, occ) & straight).any()
    }

    /// The enemy pieces currently giving check to the side to move. Empty when not in check.
    #[inline(always)]
    pub fn checkers(&self) -> Bitboard {
        let king = self.king_square(self.side);
        self.attackers_to(king, self.occupied()) & self.colors[self.side.flip().idx()]
    }

    /// Whether the side to move is in check.
    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.attacked_by(
            self.king_square(self.side),
            self.side.flip(),
            self.occupied(),
        )
    }

    /// Whether `color` has at least one knight, bishop, rook or queen.
    ///
    /// Null-move pruning is disabled when this is `false`, because pawn endings are
    /// where zugzwang (every move makes things worse) is common and passing would
    /// give misleadingly good results.
    #[inline(always)]
    pub fn has_non_pawn_material(&self, color: Color) -> bool {
        let others = self.pieces[PieceType::Pawn.idx()] | self.pieces[PieceType::King.idx()];
        (self.colors[color.idx()] & !others).any()
    }

    /// Whether neither side can ever deliver checkmate by any sequence of legal moves
    /// that this simple test recognises: no pawns, rooks or queens, and at most one
    /// minor piece on the board in total.
    ///
    /// That covers king versus king, king and bishop versus king, and king and knight
    /// versus king. Positions such as two knights versus a bare king are not
    /// reported, since mate is technically possible there.
    pub fn insufficient_material(&self) -> bool {
        let heavy = self.pieces[PieceType::Pawn.idx()]
            | self.pieces[PieceType::Rook.idx()]
            | self.pieces[PieceType::Queen.idx()];
        if heavy.any() {
            return false;
        }
        let minors = self.pieces[PieceType::Knight.idx()] | self.pieces[PieceType::Bishop.idx()];
        minors.count() <= 1
    }

    /// Adds a piece to an empty square, updating bitboards, mailbox and hash.
    #[inline(always)]
    fn put_piece(&mut self, piece: Piece, sq: Square) {
        let bit = sq.bb();
        self.pieces[piece.piece_type().idx()] |= bit;
        self.colors[piece.color().idx()] |= bit;
        self.mailbox[sq.idx()] = Some(piece);
        self.hash ^= ZOBRIST.piece_square[piece.idx()][sq.idx()];
    }

    /// Removes whatever piece is on `sq`, updating bitboards, mailbox and hash. Does nothing on an empty square.
    #[inline(always)]
    fn remove_piece(&mut self, sq: Square) {
        if let Some(piece) = self.mailbox[sq.idx()] {
            let bit = sq.bb();
            self.pieces[piece.piece_type().idx()] ^= bit;
            self.colors[piece.color().idx()] ^= bit;
            self.mailbox[sq.idx()] = None;
            self.hash ^= ZOBRIST.piece_square[piece.idx()][sq.idx()];
        }
    }

    /// Moves a piece to an empty destination with one XOR per bitboard. Captures must be removed first.
    #[inline(always)]
    fn move_piece(&mut self, from: Square, to: Square) {
        if let Some(piece) = self.mailbox[from.idx()] {
            let bits = from.bb() | to.bb();
            self.pieces[piece.piece_type().idx()] ^= bits;
            self.colors[piece.color().idx()] ^= bits;
            self.mailbox[from.idx()] = None;
            self.mailbox[to.idx()] = Some(piece);
            self.hash ^= ZOBRIST.piece_square[piece.idx()][from.idx()]
                ^ ZOBRIST.piece_square[piece.idx()][to.idx()];
        }
    }

    /// Recomputes the Zobrist hash from scratch.
    ///
    /// [`Position::hash`] is normally maintained incrementally; this slow version is
    /// the reference used by tests and by [`Position::is_consistent`] to prove the
    /// incremental updates are right.
    pub fn compute_hash(&self) -> u64 {
        let mut hash = 0u64;
        for sq in self.occupied() {
            if let Some(piece) = self.mailbox[sq.idx()] {
                hash ^= ZOBRIST.piece_square[piece.idx()][sq.idx()];
            }
        }
        hash ^= ZOBRIST.castling[self.castling as usize];
        if self.ep != NO_SQUARE {
            hash ^= ZOBRIST.ep_file[(self.ep & 7) as usize];
        }
        if self.side == Color::Black {
            hash ^= ZOBRIST.side;
        }
        hash
    }

    /// Checks the internal invariants: the piece and colour bitboards agree with the
    /// mailbox, the two colours do not overlap, and the stored hash equals
    /// [`Position::compute_hash`].
    ///
    /// Intended for tests. The test suite walks full move trees and asserts this at
    /// every node.
    pub fn is_consistent(&self) -> bool {
        let mut pieces = [Bitboard::EMPTY; 6];
        let mut colors = [Bitboard::EMPTY; 2];
        for (i, piece) in self.mailbox.iter().enumerate() {
            if let Some(piece) = piece {
                pieces[piece.piece_type().idx()] |= Square::from_idx(i).bb();
                colors[piece.color().idx()] |= Square::from_idx(i).bb();
            }
        }
        pieces == self.pieces
            && colors == self.colors
            && (self.colors[0] & self.colors[1]).is_empty()
            && self.hash == self.compute_hash()
    }

    /// Returns the position after playing `mv`. The receiver is unchanged.
    ///
    /// `mv` must be legal in this position, for example obtained from
    /// [`crate::movegen::legal_moves`]. See the [module documentation](crate::board)
    /// for exactly what is updated.
    ///
    /// ```
    /// use wustite::Position;
    /// use wustite::movegen::legal_moves;
    ///
    /// let pos = Position::startpos();
    /// for &mv in legal_moves(&pos).iter() {
    ///     let child = pos.make_move(mv);
    ///     assert!(child.is_consistent());
    ///     assert_eq!(child.side_to_move(), !pos.side_to_move());
    /// }
    /// ```
    pub fn make_move(&self, mv: Move) -> Position {
        let mut next = *self;
        next.apply_move(mv);
        next
    }

    /// The mutating core of [`Position::make_move`], applied to a fresh copy.
    fn apply_move(&mut self, mv: Move) {
        let us = self.side;
        let them = us.flip();
        let from = mv.from_sq();
        let to = mv.to_sq();
        let flags = mv.flags();
        let Some(piece) = self.mailbox[from.idx()] else {
            debug_assert!(false, "make_move from an empty square");
            return;
        };

        if self.ep != NO_SQUARE {
            self.hash ^= ZOBRIST.ep_file[(self.ep & 7) as usize];
            self.ep = NO_SQUARE;
        }

        if mv.is_capture() {
            let captured_sq = if flags == flag::EN_PASSANT {
                to.offset(if us == Color::White { -8 } else { 8 })
            } else {
                to
            };
            self.remove_piece(captured_sq);
        }

        match mv.promotion_piece() {
            Some(promoted) => {
                self.remove_piece(from);
                self.put_piece(Piece::new(us, promoted), to);
            }
            None => self.move_piece(from, to),
        }

        if flags == flag::KING_CASTLE {
            self.move_piece(to.offset(1), to.offset(-1));
        } else if flags == flag::QUEEN_CASTLE {
            self.move_piece(to.offset(-2), to.offset(1));
        }

        let new_castling = self.castling & CASTLE_MASK[from.idx()] & CASTLE_MASK[to.idx()];
        self.hash ^=
            ZOBRIST.castling[self.castling as usize] ^ ZOBRIST.castling[new_castling as usize];
        self.castling = new_castling;

        if flags == flag::DOUBLE_PUSH {
            let ep_sq = from.offset(if us == Color::White { 8 } else { -8 });
            if (pawn_attacks(us, ep_sq) & self.pieces(them, PieceType::Pawn)).any() {
                self.ep = ep_sq.0;
                self.hash ^= ZOBRIST.ep_file[ep_sq.file() as usize];
            }
        }

        if piece.piece_type() == PieceType::Pawn || mv.is_capture() {
            self.halfmove = 0;
        } else {
            self.halfmove = self.halfmove.saturating_add(1);
        }
        if us == Color::Black {
            self.fullmove = self.fullmove.saturating_add(1);
        }

        self.side = them;
        self.hash ^= ZOBRIST.side;
    }

    /// Returns the position after the side to move passes.
    ///
    /// Used by null-move pruning in the search. A null move flips the side to move,
    /// clears the en passant square, advances the halfmove clock, and updates the
    /// hash accordingly. It is only meaningful when the side to move is **not** in
    /// check.
    pub fn make_null_move(&self) -> Position {
        let mut next = *self;
        if next.ep != NO_SQUARE {
            next.hash ^= ZOBRIST.ep_file[(next.ep & 7) as usize];
            next.ep = NO_SQUARE;
        }
        next.halfmove = next.halfmove.saturating_add(1);
        next.side = next.side.flip();
        next.hash ^= ZOBRIST.side;
        next
    }

    /// Parses a FEN string. See the [module documentation](crate::board) for what is
    /// validated and what is tolerated.
    ///
    /// # Errors
    ///
    /// Returns [`FenError::MissingField`] when the piece placement or side-to-move
    /// field is missing, and [`FenError::Invalid`] for malformed or illegal content.
    ///
    /// ```
    /// use wustite::Position;
    ///
    /// let pos = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
    /// assert_eq!(pos.castling(), 15);
    ///
    /// assert!(Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1").is_err());
    /// ```
    pub fn from_fen(fen: &str) -> Result<Position, FenError> {
        let mut fields = fen.split_whitespace();
        let placement = fields
            .next()
            .ok_or(FenError::MissingField("piece placement"))?;
        let side_field = fields
            .next()
            .ok_or(FenError::MissingField("side to move"))?;
        let castling_field = fields.next().unwrap_or("-");
        let ep_field = fields.next().unwrap_or("-");
        let halfmove_field = fields.next().unwrap_or("0");
        let fullmove_field = fields.next().unwrap_or("1");

        let mut pos = Position::empty();

        let mut rank_count = 0usize;
        for (i, row) in placement.split('/').enumerate() {
            if i >= 8 {
                return Err(FenError::Invalid("more than eight ranks"));
            }
            let rank = 7 - i as u8;
            let mut file = 0u8;
            for ch in row.chars() {
                if let Some(digit) = ch.to_digit(10) {
                    if digit == 0 || digit > 8 {
                        return Err(FenError::Invalid("bad empty-square count"));
                    }
                    file += digit as u8;
                    if file > 8 {
                        return Err(FenError::Invalid("rank is wider than eight files"));
                    }
                } else if let Some(piece) = Piece::from_char(ch) {
                    if file >= 8 {
                        return Err(FenError::Invalid("rank is wider than eight files"));
                    }
                    pos.put_piece(piece, Square::new(file, rank));
                    file += 1;
                } else {
                    return Err(FenError::Invalid("unknown piece character"));
                }
            }
            if file != 8 {
                return Err(FenError::Invalid("rank does not cover eight files"));
            }
            rank_count += 1;
        }
        if rank_count != 8 {
            return Err(FenError::Invalid("fewer than eight ranks"));
        }

        pos.side = match side_field {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(FenError::Invalid("side to move must be w or b")),
        };

        for color in Color::ALL {
            if pos.pieces(color, PieceType::King).count() != 1 {
                return Err(FenError::Invalid("each side needs exactly one king"));
            }
            if pos.pieces(color, PieceType::Pawn).count() > 8 {
                return Err(FenError::Invalid("more than eight pawns"));
            }
            if pos.color_bb(color).count() > 16 {
                return Err(FenError::Invalid("more than sixteen pieces"));
            }
        }
        if (pos.piece_bb(PieceType::Pawn) & (Bitboard::RANK_1 | Bitboard::RANK_8)).any() {
            return Err(FenError::Invalid("pawn on the first or last rank"));
        }
        if pos.attacked_by(pos.king_square(pos.side.flip()), pos.side, pos.occupied()) {
            return Err(FenError::Invalid("the side not to move is in check"));
        }

        let mut rights = 0u8;
        for ch in castling_field.chars() {
            match ch {
                'K' => rights |= WHITE_KING_SIDE,
                'Q' => rights |= WHITE_QUEEN_SIDE,
                'k' => rights |= BLACK_KING_SIDE,
                'q' => rights |= BLACK_QUEEN_SIDE,
                '-' => {}
                _ => return Err(FenError::Invalid("unknown castling character")),
            }
        }
        let home = |king: Square, rook: Square, color: Color| {
            pos.piece_on(king) == Some(Piece::new(color, PieceType::King))
                && pos.piece_on(rook) == Some(Piece::new(color, PieceType::Rook))
        };
        if !home(Square::E1, Square::H1, Color::White) {
            rights &= !WHITE_KING_SIDE;
        }
        if !home(Square::E1, Square::A1, Color::White) {
            rights &= !WHITE_QUEEN_SIDE;
        }
        if !home(Square::E8, Square::H8, Color::Black) {
            rights &= !BLACK_KING_SIDE;
        }
        if !home(Square::E8, Square::A8, Color::Black) {
            rights &= !BLACK_QUEEN_SIDE;
        }
        pos.castling = rights;

        if ep_field != "-" {
            let ep_sq =
                Square::parse(ep_field).ok_or(FenError::Invalid("bad en passant square"))?;
            let us = pos.side;
            let them = us.flip();
            let expected_rank = if us == Color::White { 5 } else { 2 };
            if ep_sq.rank() != expected_rank {
                return Err(FenError::Invalid("en passant square on the wrong rank"));
            }
            let pawn_sq = ep_sq.offset(if us == Color::White { -8 } else { 8 });
            let origin_sq = ep_sq.offset(if us == Color::White { 8 } else { -8 });
            let playable = pos.piece_on(pawn_sq) == Some(Piece::new(them, PieceType::Pawn))
                && pos.piece_on(ep_sq).is_none()
                && pos.piece_on(origin_sq).is_none()
                && (pawn_attacks(them, ep_sq) & pos.pieces(us, PieceType::Pawn)).any();
            if playable {
                pos.ep = ep_sq.0;
            }
        }

        pos.halfmove = halfmove_field
            .parse()
            .map_err(|_| FenError::Invalid("bad halfmove clock"))?;
        pos.fullmove = fullmove_field
            .parse::<u16>()
            .map_err(|_| FenError::Invalid("bad fullmove number"))?
            .max(1);

        pos.hash = pos.compute_hash();
        Ok(pos)
    }

    /// Formats the position as a FEN string. Parsing the result with
    /// [`Position::from_fen`] yields an equal position.
    pub fn to_fen(&self) -> String {
        let mut out = String::with_capacity(90);
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                match self.mailbox[Square::new(file, rank).idx()] {
                    Some(piece) => {
                        if empty > 0 {
                            out.push((b'0' + empty) as char);
                            empty = 0;
                        }
                        out.push(piece.to_char());
                    }
                    None => empty += 1,
                }
            }
            if empty > 0 {
                out.push((b'0' + empty) as char);
            }
            if rank > 0 {
                out.push('/');
            }
        }
        out.push(' ');
        out.push(if self.side == Color::White { 'w' } else { 'b' });
        out.push(' ');
        if self.castling == 0 {
            out.push('-');
        } else {
            for (bit, ch) in [
                (WHITE_KING_SIDE, 'K'),
                (WHITE_QUEEN_SIDE, 'Q'),
                (BLACK_KING_SIDE, 'k'),
                (BLACK_QUEEN_SIDE, 'q'),
            ] {
                if self.castling & bit != 0 {
                    out.push(ch);
                }
            }
        }
        out.push(' ');
        match self.ep_square() {
            Some(sq) => out.push_str(&sq.to_string()),
            None => out.push('-'),
        }
        out.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        out
    }
}

impl FromStr for Position {
    type Err = FenError;

    fn from_str(s: &str) -> Result<Position, FenError> {
        Position::from_fen(s)
    }
}

impl Default for Position {
    fn default() -> Position {
        Position::startpos()
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            write!(f, "{} ", rank + 1)?;
            for file in 0..8 {
                let ch = match self.mailbox[Square::new(file, rank).idx()] {
                    Some(piece) => piece.to_char(),
                    None => '.',
                };
                write!(f, "{ch} ")?;
            }
            writeln!(f)?;
        }
        writeln!(f, "  a b c d e f g h")?;
        writeln!(f, "Fen: {}", self.to_fen())?;
        write!(f, "Key: {:016x}", self.hash)
    }
}
