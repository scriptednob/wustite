//! Fully legal move generation.
//!
//! [`generate`] produces only legal moves. There is no pseudo-legal pass followed
//! by a make-and-test filter: legality is built into the generation, so every
//! move written to the list can be played without further checks. That removes
//! the most expensive part of a typical pseudo-legal generator and keeps the
//! search loop free of "was that move legal?" branches.
//!
//! # How legality is enforced
//!
//! 1. **King moves** are generated first. A destination is kept only if no enemy
//!    piece attacks it, evaluated with the king removed from the occupancy so
//!    that a slider's ray "sees through" the king's current square.
//! 2. **Double check** ends generation: only king moves can answer it.
//! 3. **Pins** are found by looking at enemy sliders that would attack the king
//!    on an empty board. If exactly one piece stands between such a slider and the
//!    king, and it is ours, it is pinned. Pinned pieces may only move along the
//!    line through the king and the pinned piece ([`crate::attacks::line`]).
//! 4. **Single check**: every non-king move must either capture the checker or
//!    land on a square between the checker and the king
//!    ([`crate::attacks::between`]). This set is the `target` mask. When not in
//!    check the mask is the whole board.
//! 5. **En passant** is the one move whose legality cannot be decided by pin
//!    masks, because it removes two pawns from the same rank at once and can
//!    expose the king horizontally. It is verified by recomputing the attackers
//!    of the king on the post-capture occupancy.
//! 6. **Castling** requires the right, empty squares between king and rook, and
//!    that the king does not pass through attacked squares. It is never generated
//!    while in check.
//!
//! # Move-generation stages
//!
//! [`generate`] is generic over a `const MODE: u8`, which the compiler turns into
//! three separate specialised functions with the unused branches removed:
//!
//! | Mode | Constant | Generates |
//! | --- | --- | --- |
//! | all legal moves | [`GEN_ALL`] | captures, promotions and quiet moves |
//! | tactical moves | [`GEN_TACTICAL`] | captures, en passant and promotions |
//! | quiet moves | [`GEN_QUIET`] | non-captures (including castling and double pushes) |
//!
//! Quiescence search uses `GEN_TACTICAL`. When in check it needs every evasion,
//! so it uses `GEN_ALL` instead. Quiet promotions are classed as tactical.
//!
//! # Fixed-size output
//!
//! [`MoveList`] is a stack-allocated array of [`MAX_MOVES`] moves plus a length.
//! There are no heap allocations anywhere in move generation. The largest known
//! number of legal moves in a reachable position is 218, so 256 is a safe bound.
//!
//! # Examples
//!
//! ```
//! use wustite::Position;
//! use wustite::movegen::{generate, legal_moves, GEN_QUIET, GEN_TACTICAL};
//! use wustite::MoveList;
//!
//! let pos: Position =
//!     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1".parse().unwrap();
//!
//! let all = legal_moves(&pos);
//! assert_eq!(all.len(), 48);
//!
//! let mut tactical = MoveList::new();
//! generate::<GEN_TACTICAL>(&pos, &mut tactical);
//! let mut quiet = MoveList::new();
//! generate::<GEN_QUIET>(&pos, &mut quiet);
//! assert_eq!(tactical.len() + quiet.len(), all.len());
//! ```

use core::ops::{Deref, DerefMut};

use crate::attacks::{attacks_bb, between, bishop_attacks, line, pawn_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::board::{
    BLACK_KING_SIDE, BLACK_QUEEN_SIDE, Position, WHITE_KING_SIDE, WHITE_QUEEN_SIDE,
};
use crate::types::{Color, Move, PieceType, Square, flag};

/// Capacity of a [`MoveList`]. No reachable chess position has more than 218 legal moves.
pub const MAX_MOVES: usize = 256;

/// [`generate`] mode: every legal move.
pub const GEN_ALL: u8 = 0;
/// [`generate`] mode: captures, en passant captures and all promotions.
pub const GEN_TACTICAL: u8 = 1;
/// [`generate`] mode: legal moves that are neither captures nor promotions (including castling and pawn pushes).
pub const GEN_QUIET: u8 = 2;

/// A fixed-capacity, stack-allocated list of moves.
///
/// `MoveList` dereferences to `[Move]`, so all slice methods work: `len`, `iter`,
/// `swap`, indexing, `contains`, sorting and so on. Only [`MoveList::push`] and
/// [`MoveList::clear`] change the length.
///
/// The backing array is [`MAX_MOVES`] entries, about 512 bytes, which is cheap to
/// create at every search node.
///
/// ```
/// use wustite::{Move, MoveList, Square};
/// use wustite::types::flag;
///
/// let mut list = MoveList::new();
/// assert!(list.is_empty());
/// list.push(Move::new(Square::E2, Square::E4, flag::DOUBLE_PUSH));
/// assert_eq!(list.len(), 1);
/// assert_eq!(list[0].to_string(), "e2e4");
/// list.clear();
/// assert!(list.is_empty());
/// ```
#[derive(Clone)]
pub struct MoveList {
    moves: [Move; MAX_MOVES],
    len: usize,
}

impl MoveList {
    /// An empty list.
    #[inline(always)]
    pub const fn new() -> MoveList {
        MoveList {
            moves: [Move::NONE; MAX_MOVES],
            len: 0,
        }
    }

    /// Appends a move.
    ///
    /// # Panics
    ///
    /// Panics if the list already holds [`MAX_MOVES`] moves, which cannot happen with
    /// moves from [`generate`].
    #[inline(always)]
    pub fn push(&mut self, mv: Move) {
        self.moves[self.len] = mv;
        self.len += 1;
    }

    /// Empties the list without touching the backing array.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

impl Default for MoveList {
    fn default() -> MoveList {
        MoveList::new()
    }
}

impl Deref for MoveList {
    type Target = [Move];

    #[inline(always)]
    fn deref(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

impl DerefMut for MoveList {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [Move] {
        &mut self.moves[..self.len]
    }
}

/// Every legal move in `pos`. Shorthand for [`generate`] with [`GEN_ALL`].
///
/// An empty result means checkmate (when [`Position::in_check`]) or stalemate.
///
/// ```
/// use wustite::Position;
/// use wustite::movegen::legal_moves;
///
/// assert_eq!(legal_moves(&Position::startpos()).len(), 20);
///
/// let mated: Position = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1".parse().unwrap();
/// assert!(legal_moves(&mated).is_empty() && mated.in_check());
/// ```
pub fn legal_moves(pos: &Position) -> MoveList {
    let mut list = MoveList::new();
    generate::<GEN_ALL>(pos, &mut list);
    list
}

/// Finds the legal move whose UCI text equals `text`, such as `"e2e4"` or `"a7a8q"`.
///
/// Whitespace around the text is ignored. Returns `None` when no legal move in
/// `pos` has that notation, so this doubles as a legality check for GUI input.
/// Castling uses the king's two-square move (`e1g1`), as in standard UCI.
///
/// ```
/// use wustite::Position;
/// use wustite::movegen::parse_move;
///
/// let pos = Position::startpos();
/// assert!(parse_move(&pos, "g1f3").is_some());
/// assert!(parse_move(&pos, "e2e5").is_none());
/// ```
pub fn parse_move(pos: &Position, text: &str) -> Option<Move> {
    let text = text.trim();
    legal_moves(pos)
        .iter()
        .copied()
        .find(|mv| mv.to_string() == text)
}

/// Pushes one move per destination in `targets`, all from the same origin with the same flags.
#[inline(always)]
fn emit(list: &mut MoveList, from: Square, targets: Bitboard, flags: u16) {
    for to in targets {
        list.push(Move::new(from, to, flags));
    }
}

/// Pushes pawn moves for a whole target set at once. The origin of each move is recovered by subtracting `delta` from its destination.
#[inline(always)]
fn emit_pawn_moves(list: &mut MoveList, targets: Bitboard, delta: i8, flags: u16) {
    for to in targets {
        list.push(Move::new(to.offset(-delta), to, flags));
    }
}

/// Pushes the four promotion moves (queen, rook, bishop, knight) for every target square, as captures when `capture` is set.
#[inline(always)]
fn emit_promotions(list: &mut MoveList, targets: Bitboard, delta: i8, capture: bool) {
    let extra = if capture { flag::PROMO_CAPTURE_BIT } else { 0 };
    for to in targets {
        let from = to.offset(-delta);
        list.push(Move::new(from, to, flag::PROMO_QUEEN | extra));
        list.push(Move::new(from, to, flag::PROMO_ROOK | extra));
        list.push(Move::new(from, to, flag::PROMO_BISHOP | extra));
        list.push(Move::new(from, to, flag::PROMO_KNIGHT | extra));
    }
}

/// Appends the legal moves of `pos` selected by `MODE` to `list`.
///
/// `MODE` is one of [`GEN_ALL`], [`GEN_TACTICAL`] or [`GEN_QUIET`]. The list is
/// **not** cleared first, so callers can accumulate several stages. See the
/// [module documentation](crate::movegen) for how legality is guaranteed.
///
/// Generation order is: king moves, then knights, bishops, rooks and queens,
/// then pawns, en passant and castling. Callers that care about order (the
/// search) re-sort by score.
///
/// ```
/// use wustite::{MoveList, Position};
/// use wustite::movegen::{generate, GEN_ALL};
///
/// let pos = Position::startpos();
/// let mut list = MoveList::new();
/// generate::<GEN_ALL>(&pos, &mut list);
/// assert_eq!(list.len(), 20);
/// ```
pub fn generate<const MODE: u8>(pos: &Position, list: &mut MoveList) {
    let us = pos.side_to_move();
    let them = us.flip();
    let king = pos.king_square(us);
    let occ = pos.occupied();
    let ours = pos.color_bb(us);
    let theirs = pos.color_bb(them);
    let tactical = MODE != GEN_QUIET;
    let quiet = MODE != GEN_TACTICAL;
    let checkers = pos.checkers();

    let occ_without_king = occ ^ king.bb();
    let mut king_targets = crate::attacks::king_attacks(king) & !ours;
    if !tactical {
        king_targets &= !theirs;
    }
    if !quiet {
        king_targets &= theirs;
    }
    for to in king_targets {
        if !pos.attacked_by(to, them, occ_without_king) {
            let flags = if theirs.contains(to) {
                flag::CAPTURE
            } else {
                flag::QUIET
            };
            list.push(Move::new(king, to, flags));
        }
    }

    if checkers.more_than_one() {
        return;
    }

    let diagonal_snipers = (pos.pieces(them, PieceType::Bishop)
        | pos.pieces(them, PieceType::Queen))
        & bishop_attacks(king, Bitboard::EMPTY);
    let straight_snipers = (pos.pieces(them, PieceType::Rook) | pos.pieces(them, PieceType::Queen))
        & rook_attacks(king, Bitboard::EMPTY);
    let mut pinned = Bitboard::EMPTY;
    for sniper in diagonal_snipers | straight_snipers {
        let blockers = between(king, sniper) & occ;
        if blockers.any() && !blockers.more_than_one() {
            pinned |= blockers & ours;
        }
    }

    let target = if checkers.any() {
        between(king, checkers.lsb()) | checkers
    } else {
        Bitboard::FULL
    };
    let capture_mask = if tactical {
        theirs & target
    } else {
        Bitboard::EMPTY
    };
    let quiet_mask = if quiet {
        !occ & target
    } else {
        Bitboard::EMPTY
    };

    for piece_type in [
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ] {
        for from in pos.pieces(us, piece_type) {
            let mut destinations = attacks_bb(piece_type, from, occ);
            if pinned.contains(from) {
                destinations &= line(king, from);
            }
            emit(list, from, destinations & capture_mask, flag::CAPTURE);
            emit(list, from, destinations & quiet_mask, flag::QUIET);
        }
    }

    let pawns = pos.pieces(us, PieceType::Pawn);
    generate_pawns::<MODE>(us, pawns & !pinned, target, occ, theirs, list);
    for pawn in pawns & pinned {
        generate_pawns::<MODE>(us, pawn.bb(), target & line(king, pawn), occ, theirs, list);
    }

    if tactical {
        generate_en_passant(pos, us, king, occ, list);
    }
    if quiet && checkers.is_empty() {
        generate_castling(pos, us, occ, list);
    }
}

/// Pawn pushes, double pushes, captures and promotions for a set of pawns,
/// restricted to the destination mask `target` (check evasion and pin masks).
///
/// Pushes and captures are computed for the whole set with shifts rather than one
/// pawn at a time. Pinned pawns are handled by calling this once per pawn with
/// the target restricted to the pin line.
#[inline(always)]
fn generate_pawns<const MODE: u8>(
    us: Color,
    pawns: Bitboard,
    target: Bitboard,
    occ: Bitboard,
    theirs: Bitboard,
    list: &mut MoveList,
) {
    if pawns.is_empty() {
        return;
    }
    let tactical = MODE != GEN_QUIET;
    let quiet = MODE != GEN_TACTICAL;
    let empty = !occ;

    let (up, left, right, third, last, forward, capture_left, capture_right) = if us == Color::White
    {
        (
            8i8,
            7i8,
            9i8,
            Bitboard::RANK_3,
            Bitboard::RANK_8,
            pawns.north(),
            pawns.north_west(),
            pawns.north_east(),
        )
    } else {
        (
            -8i8,
            -9i8,
            -7i8,
            Bitboard::RANK_6,
            Bitboard::RANK_1,
            pawns.south(),
            pawns.south_west(),
            pawns.south_east(),
        )
    };

    let single = forward & empty;
    let double = if us == Color::White {
        (single & third).north() & empty
    } else {
        (single & third).south() & empty
    };

    if quiet {
        emit_pawn_moves(list, single & target & !last, up, flag::QUIET);
        emit_pawn_moves(list, double & target, 2 * up, flag::DOUBLE_PUSH);
    }

    if tactical {
        emit_promotions(list, single & target & last, up, false);

        let left_captures = capture_left & theirs & target;
        let right_captures = capture_right & theirs & target;
        emit_pawn_moves(list, left_captures & !last, left, flag::CAPTURE);
        emit_pawn_moves(list, right_captures & !last, right, flag::CAPTURE);
        emit_promotions(list, left_captures & last, left, true);
        emit_promotions(list, right_captures & last, right, true);
    }
}

/// En passant captures, verified by recomputing king safety on the occupancy
/// after both pawns have left their squares. That catches the rare "discovered
/// rank pin" where capturing removes two blockers between a rook or queen and the
/// king.
fn generate_en_passant(
    pos: &Position,
    us: Color,
    king: Square,
    occ: Bitboard,
    list: &mut MoveList,
) {
    let Some(ep) = pos.ep_square() else {
        return;
    };
    let them = us.flip();
    let captured = ep.offset(if us == Color::White { -8 } else { 8 });
    let theirs = pos.color_bb(them);
    for from in pawn_attacks(them, ep) & pos.pieces(us, PieceType::Pawn) {
        let occ_after = (occ ^ from.bb() ^ captured.bb()) | ep.bb();
        let attackers = pos.attackers_to(king, occ_after) & theirs & !captured.bb();
        if attackers.is_empty() {
            list.push(Move::new(from, ep, flag::EN_PASSANT));
        }
    }
}

/// Castling moves for `us`, when the right exists, the squares between king and
/// rook are empty and the king's start, transit and destination squares are not
/// attacked. Never called while in check.
fn generate_castling(pos: &Position, us: Color, occ: Bitboard, list: &mut MoveList) {
    let them = us.flip();
    let rights = pos.castling();
    let (king_side, queen_side, rank) = if us == Color::White {
        (WHITE_KING_SIDE, WHITE_QUEEN_SIDE, 0)
    } else {
        (BLACK_KING_SIDE, BLACK_QUEEN_SIDE, 7)
    };
    let king = Square::new(4, rank);

    if rights & king_side != 0 {
        let f = Square::new(5, rank);
        let g = Square::new(6, rank);
        if (occ & (f.bb() | g.bb())).is_empty()
            && !pos.attacked_by(f, them, occ)
            && !pos.attacked_by(g, them, occ)
        {
            list.push(Move::new(king, g, flag::KING_CASTLE));
        }
    }

    if rights & queen_side != 0 {
        let b = Square::new(1, rank);
        let c = Square::new(2, rank);
        let d = Square::new(3, rank);
        if (occ & (b.bb() | c.bb() | d.bb())).is_empty()
            && !pos.attacked_by(d, them, occ)
            && !pos.attacked_by(c, them, occ)
        {
            list.push(Move::new(king, c, flag::QUEEN_CASTLE));
        }
    }
}
