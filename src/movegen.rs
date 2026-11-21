//! Fully legal, allocation-free move generation.
//!
//! [`generate`] only writes legal moves into a stack-allocated [`MoveList`], so
//! the search never has to make-and-reject illegal pseudo-legal moves.
//!
//! # How legality is checked up front
//!
//! 1. King moves go first, testing each target against enemy attacks with the
//!    king removed from occupancy (so sliding checks don't get blocked by the
//!    king's own square).
//! 2. In double check, return right away—only king moves can get out of it.
//! 3. Detect pinned pieces by finding enemy sliders aligned with our king that
//!    have a single friendly blocker in between, and restrict those pieces to
//!    [`crate::attacks::line`].
//! 4. In single check, restrict non-king moves to capturing the checker or
//!    blocking on [`crate::attacks::between`].
//! 5. En passant verifies king safety on the post-capture occupancy directly,
//!    catching the rare horizontal discovered pin when both pawns vanish from
//!    the same rank.
//! 6. Castling checks rights, empty squares, and non-attacked king transit
//!    squares (skipped completely while in check).
//!
//! `const MODE: u8` lets the compiler emit three stripped-down variants:
//! [`GEN_ALL`], [`GEN_TACTICAL`] (captures, en passant, and all promotions),
//! and [`GEN_QUIET`].
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

/// Fixed capacity of [`MoveList`] (the max legal moves in any reachable position is 218).
pub const MAX_MOVES: usize = 256;

/// Generate every legal move.
pub const GEN_ALL: u8 = 0;
/// Generate captures, en passant, and all promotions (including quiet promotions).
pub const GEN_TACTICAL: u8 = 1;
/// Generate non-capturing, non-promoting moves (including castling and pawn pushes).
pub const GEN_QUIET: u8 = 2;

/// Stack-allocated move buffer (capacity [`MAX_MOVES`], ~512 bytes).
///
/// Derefs to `[Move]`, so indexing, slicing, `iter`, `swap`, and `contains` all
/// work out of the box.
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
    /// Create an empty move list.
    #[inline(always)]
    pub const fn new() -> MoveList {
        MoveList {
            moves: [Move::NONE; MAX_MOVES],
            len: 0,
        }
    }

    /// Append a move. Panics if the list already has [`MAX_MOVES`] entries.
    #[inline(always)]
    pub fn push(&mut self, mv: Move) {
        self.moves[self.len] = mv;
        self.len += 1;
    }

    /// Reset the length to zero without zeroing the array.
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

/// Generate and return all legal moves in `pos`.
///
/// An empty list means checkmate (if `pos.in_check()`) or stalemate.
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

/// Match a UCI move string (like `"e2e4"` or `"a7a8q"`) against the legal moves in `pos`.
///
/// Returns `None` if the move is malformed or illegal, so UCI input can use
/// this directly for validation.
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

/// Push one move per bit in `targets` sharing `from` and `flags`.
#[inline(always)]
fn emit(list: &mut MoveList, from: Square, targets: Bitboard, flags: u16) {
    for to in targets {
        list.push(Move::new(from, to, flags));
    }
}

/// Push shifted pawn moves, recovering each origin square as `to - delta`.
#[inline(always)]
fn emit_pawn_moves(list: &mut MoveList, targets: Bitboard, delta: i8, flags: u16) {
    for to in targets {
        list.push(Move::new(to.offset(-delta), to, flags));
    }
}

/// Push all four promotion choices (Q, R, B, N) for every square in `targets`.
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

/// Append legal moves matching `MODE` (`GEN_ALL`, `GEN_TACTICAL`, or `GEN_QUIET`) to `list`.
///
/// Does not clear `list` first so callers can combine stages if needed.
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

/// Emit bulk-shifted pawn pushes, double pushes, captures, and promotions clipped to `target`.
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

/// Emit legal en passant captures, checking post-capture occupancy for horizontal discovered pins.
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

/// Emit legal kingside/queenside castling moves. Caller ensures we aren't in check.
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
