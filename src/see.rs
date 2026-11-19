//! Static Exchange Evaluation (SEE).
//!
//! SEE answers: "if both sides keep capturing on this square, always using their
//! least valuable attacker and stopping whenever stopping is better, does the side
//! that starts the exchange come out ahead?" It does this without searching, by
//! simulating the sequence of recaptures on one square.
//!
//! [`see_ge`] does not return the exchange value. It answers the boolean
//! question "is the outcome at least `threshold`?", which is all the search needs
//! and allows early exits as soon as the answer is certain.
//!
//! # Algorithm
//!
//! This is the *swap* algorithm, in the same formulation used by strong open
//! engines:
//!
//! 1. Start with `swap = captured value - threshold`. If that is already negative
//!    the capture cannot reach the threshold even if it is never recaptured.
//! 2. Then `swap = attacker value - swap`. If that is `<= 0`, the capture reaches
//!    the threshold even if the capturing piece is immediately lost, so the
//!    answer is yes.
//! 3. Remove the moving piece from the occupancy and gather all attackers of the
//!    target square. Alternate sides; each side picks its least valuable attacker
//!    (pawn, knight, bishop, rook, queen, then king). After each capture,
//!    sliders behind the removed piece are added to the attacker set (x-rays).
//! 4. A `result` flag toggles each time a side has a legal recapture; the loop
//!    ends when a side has none left, or when continuing cannot help it.
//!
//! # Pins
//!
//! A piece that is pinned to its king cannot recapture off the pin line. Plain
//! SEE ignores this and overestimates defenders. Here the pinned and pinning
//! pieces of both sides are computed once up front, and while the pinning piece is
//! still on the board, pinned pieces are removed from the candidate attackers.
//!
//! # Where it is used
//!
//! * **Move ordering**: captures with SEE >= 0 are "good" and searched before
//!   quiet moves; the rest are "bad captures" and come last.
//! * **Quiescence search**: captures that lose material by SEE are skipped.
//! * **Pruning in the main search**: at shallow depth, moves whose SEE is below
//!   a depth-scaled margin are not searched.
//! * **ProbCut**: only captures that SEE says can beat the cut-off are tried.
//!
//! # Simplifications
//!
//! En passant, castling and promotion moves are not evaluated as exchanges:
//! `see_ge` returns `0 >= threshold` for them. Piece values are the SEE-specific
//! [`SEE_VALUE`] table, with a very large king value so that the king never
//! "wins" a capture that would leave it en prise.
//!
//! # Examples
//!
//! ```
//! use wustite::Position;
//! use wustite::movegen::parse_move;
//! use wustite::see::see_ge;
//!
//! let pos: Position = "1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1".parse().unwrap();
//! let rook_takes_pawn = parse_move(&pos, "e1e5").unwrap();
//! assert!(see_ge(&pos, rook_takes_pawn, 0));
//! assert!(!see_ge(&pos, rook_takes_pawn, 200));
//!
//! let pawn_defended: Position = "4k3/8/3p4/4p3/8/8/4R3/4K3 w - - 0 1".parse().unwrap();
//! let rook_takes = parse_move(&pawn_defended, "e2e5").unwrap();
//! assert!(!see_ge(&pawn_defended, rook_takes, 0));
//! ```

use crate::attacks::{between, bishop_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::board::Position;
use crate::types::{Color, Move, PieceType};

/// Piece values used by SEE, in centipawns: pawn 100, knight 320, bishop 330,
/// rook 500, queen 900, king 20,000.
///
/// The king's value is deliberately huge so that capturing with the king is only
/// chosen when the square has no remaining enemy attackers. Indexed by
/// [`crate::types::PieceType::idx`].
pub const SEE_VALUE: [i32; 6] = [100, 320, 330, 500, 900, 20_000];

/// Returns `(pinned, pinners)` for `color`: the pieces of either colour standing
/// alone between an enemy slider and `color`'s king, restricted to cases where one
/// of `color`'s own pieces is the blocker, and the enemy sliders doing the
/// pinning.
fn pin_info(pos: &Position, color: Color) -> (Bitboard, Bitboard) {
    let king = pos.king_square(color);
    let them = !color;
    let occ = pos.occupied();
    let snipers = ((pos.pieces(them, PieceType::Bishop) | pos.pieces(them, PieceType::Queen))
        & bishop_attacks(king, Bitboard::EMPTY))
        | ((pos.pieces(them, PieceType::Rook) | pos.pieces(them, PieceType::Queen))
            & rook_attacks(king, Bitboard::EMPTY));
    let mut pinned = Bitboard::EMPTY;
    let mut pinners = Bitboard::EMPTY;
    for sniper in snipers {
        let blockers = between(king, sniper) & occ;
        if blockers.any() && !blockers.more_than_one() && (blockers & pos.color_bb(color)).any() {
            pinned |= blockers;
            pinners |= sniper.bb();
        }
    }
    (pinned, pinners)
}

/// Whether the exchange started by `mv` is worth at least `threshold` centipawns
/// to the side making it.
///
/// `threshold = 0` asks "does this capture at least break even?"; a positive
/// threshold asks for a material win of that size, and a negative one tolerates
/// a loss. Quiet moves are evaluated too (the captured value is zero), which the
/// search uses to avoid moving a piece onto a square where it is simply lost.
///
/// En passant, castling and promotions return `0 >= threshold` without further
/// analysis. See the [module documentation](crate::see) for the algorithm.
pub fn see_ge(pos: &Position, mv: Move, threshold: i32) -> bool {
    if mv.is_castle() || mv.is_en_passant() || mv.is_promotion() {
        return 0 >= threshold;
    }

    let from = mv.from_sq();
    let to = mv.to_sq();
    let Some(attacker) = pos.piece_on(from) else {
        return true;
    };

    let captured = pos
        .piece_on(to)
        .map_or(0, |p| SEE_VALUE[p.piece_type().idx()]);
    let mut swap = captured - threshold;
    if swap < 0 {
        return false;
    }
    swap = SEE_VALUE[attacker.piece_type().idx()] - swap;
    if swap <= 0 {
        return true;
    }

    let mut occ = pos.occupied() ^ from.bb() ^ to.bb();
    let mut side = pos.side_to_move();
    let mut attackers = pos.attackers_to(to, occ);
    let diagonal = pos.piece_bb(PieceType::Bishop) | pos.piece_bb(PieceType::Queen);
    let straight = pos.piece_bb(PieceType::Rook) | pos.piece_bb(PieceType::Queen);
    let mut result = 1i32;
    let pins = [pin_info(pos, Color::White), pin_info(pos, Color::Black)];

    loop {
        side = !side;
        attackers &= occ;
        let mut mine = attackers & pos.color_bb(side);
        let (pinned, pinners) = pins[side.idx()];
        if (pinners & occ).any() {
            mine &= !pinned;
        }
        if mine.is_empty() {
            break;
        }
        result ^= 1;

        let pawns = mine & pos.piece_bb(PieceType::Pawn);
        let knights = mine & pos.piece_bb(PieceType::Knight);
        let bishops = mine & pos.piece_bb(PieceType::Bishop);
        let rooks = mine & pos.piece_bb(PieceType::Rook);
        let queens = mine & pos.piece_bb(PieceType::Queen);

        if pawns.any() {
            swap = SEE_VALUE[PieceType::Pawn.idx()] - swap;
            if swap < result {
                break;
            }
            occ ^= pawns.lsb().bb();
            attackers |= bishop_attacks(to, occ) & diagonal;
        } else if knights.any() {
            swap = SEE_VALUE[PieceType::Knight.idx()] - swap;
            if swap < result {
                break;
            }
            occ ^= knights.lsb().bb();
        } else if bishops.any() {
            swap = SEE_VALUE[PieceType::Bishop.idx()] - swap;
            if swap < result {
                break;
            }
            occ ^= bishops.lsb().bb();
            attackers |= bishop_attacks(to, occ) & diagonal;
        } else if rooks.any() {
            swap = SEE_VALUE[PieceType::Rook.idx()] - swap;
            if swap < result {
                break;
            }
            occ ^= rooks.lsb().bb();
            attackers |= rook_attacks(to, occ) & straight;
        } else if queens.any() {
            swap = SEE_VALUE[PieceType::Queen.idx()] - swap;
            if swap < result {
                break;
            }
            occ ^= queens.lsb().bb();
            attackers |= (bishop_attacks(to, occ) & diagonal) | (rook_attacks(to, occ) & straight);
        } else {
            let opponent_left = (attackers & !pos.color_bb(side)).any();
            return if opponent_left {
                (result ^ 1) != 0
            } else {
                result != 0
            };
        }
    }

    result != 0
}
