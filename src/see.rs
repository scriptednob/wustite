//! Static Exchange Evaluation (SEE).
//!
//! Simulates a chain of recaptures on a single square without running a search,
//! always picking each side's cheapest attacker and stopping if recapturing
//! would lose material.
//!
//! Instead of computing an exact centipawn value, [`see_ge`] only answers
//! whether the exchange meets or beats `threshold`. That's all move ordering,
//! quiescence, and pruning ever ask, and it lets us bail out early as soon as
//! the result is decided.
//!
//! # How it works
//!
//! Uses the standard swap loop:
//!
//! 1. Start with `swap = captured - threshold`. If that's already negative, we
//!    can't reach the threshold even for free.
//! 2. Update `swap = attacker - swap`. If that's `<= 0`, we already clear the
//!    threshold even if the attacker is lost right away.
//! 3. Remove the moving piece from occupancy and alternate recaptures from
//!    cheapest piece to most expensive, revealing any x-ray sliders behind each
//!    removed piece as we go.
//!
//! Pinned pieces and their pinners are computed once up front for both sides.
//! As long as a side's pinner is still on the board, its pinned pieces are
//! masked out of the candidate attackers so SEE doesn't count illegal defenders.
//!
//! Castling, en passant, and promotions aren't simulated here and just return
//! `0 >= threshold`.
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

/// Piece values in centipawns used by SEE (pawn through king), indexed by [`PieceType::idx`].
///
/// The king gets 20,000 so it's only used to recapture when no opponent attacker
/// is left on the square.
pub const SEE_VALUE: [i32; 6] = [100, 320, 330, 500, 900, 20_000];

/// Find `(pinned, pinners)` for `color`: friendly single blockers and the enemy sliders pinning them.
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

/// Return `true` if the exchange triggered by `mv` scores at least `threshold` centipawns.
///
/// Works on quiet moves too (captured value is 0), which search uses to spot
/// pieces stepping onto undefended squares. Castling, en passant, and
/// promotions skip simulation and return `0 >= threshold`.
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
