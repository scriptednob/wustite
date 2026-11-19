//! Perft: the correctness test for move generation.
//!
//! *Perft* ("performance test") counts the leaf nodes of the legal-move tree to a
//! fixed depth. Because the number of legal move sequences from a position is a
//! precisely known quantity (published for many positions), any bug in move
//! generation, make-move, castling rights or en passant handling shows up as a
//! wrong count. It is the standard first test for any chess engine.
//!
//! [`SUITE`] contains the standard reference positions with known counts, and
//! [`perft`] and [`divide`] compute them. The same code backs the command-line
//! `wustite perft`, `wustite divide` and `wustite suite`, and the UCI command
//! `go perft N`.
//!
//! Move generation here is *bulk counted* at the last ply: when `depth == 1` the
//! function returns the length of the move list without making any moves, which
//! roughly doubles the speed.
//!
//! # Examples
//!
//! ```
//! use wustite::Position;
//! use wustite::perft::perft;
//!
//! let pos = Position::startpos();
//! assert_eq!(perft(&pos, 1), 20);
//! assert_eq!(perft(&pos, 2), 400);
//! assert_eq!(perft(&pos, 3), 8_902);
//! assert_eq!(perft(&pos, 4), 197_281);
//! ```
//!
//! When a count disagrees with a reference, run [`divide`] on the position and
//! compare per-move counts against another engine to find the offending move,
//! then repeat one ply deeper on that move's child position.
//!
//! ```
//! use wustite::Position;
//! use wustite::perft::divide;
//!
//! let rows = divide(&Position::startpos(), 2);
//! assert_eq!(rows.len(), 20);
//! assert!(rows.iter().all(|&(_, nodes)| nodes == 20));
//! ```

use crate::board::Position;
use crate::movegen::{GEN_ALL, MoveList, generate};
use crate::types::Move;

/// One reference position with its known node counts.
pub struct PerftCase {
    /// A short human-readable name, such as `"kiwipete"`.
    pub name: &'static str,
    /// The position in FEN.
    pub fen: &'static str,
    /// Pairs of `(depth, expected node count)`.
    pub results: &'static [(u32, u64)],
}

/// The reference perft positions.
///
/// The set covers the starting position, "Kiwipete" (a position dense with
/// tactics, castling and en passant possibilities), and the other standard
/// edge-case positions from the Chess Programming Wiki, including promotions,
/// discovered checks and en passant pins, and a colour-mirrored copy of one of
/// them. Counts up to depth 6 are listed where they are fast enough to run in a
/// test.
pub static SUITE: &[PerftCase] = &[
    PerftCase {
        name: "startpos",
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        results: &[
            (1, 20),
            (2, 400),
            (3, 8_902),
            (4, 197_281),
            (5, 4_865_609),
            (6, 119_060_324),
        ],
    },
    PerftCase {
        name: "kiwipete",
        fen: "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        results: &[
            (1, 48),
            (2, 2_039),
            (3, 97_862),
            (4, 4_085_603),
            (5, 193_690_690),
        ],
    },
    PerftCase {
        name: "position 3",
        fen: "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        results: &[
            (1, 14),
            (2, 191),
            (3, 2_812),
            (4, 43_238),
            (5, 674_624),
            (6, 11_030_083),
        ],
    },
    PerftCase {
        name: "position 4",
        fen: "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        results: &[(1, 6), (2, 264), (3, 9_467), (4, 422_333), (5, 15_833_292)],
    },
    PerftCase {
        name: "position 4 mirrored",
        fen: "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        results: &[(1, 6), (2, 264), (3, 9_467), (4, 422_333), (5, 15_833_292)],
    },
    PerftCase {
        name: "position 5",
        fen: "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        results: &[
            (1, 44),
            (2, 1_486),
            (3, 62_379),
            (4, 2_103_487),
            (5, 89_941_194),
        ],
    },
    PerftCase {
        name: "position 6",
        fen: "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        results: &[
            (1, 46),
            (2, 2_079),
            (3, 89_890),
            (4, 3_894_594),
            (5, 164_075_551),
        ],
    },
];

/// Counts the leaf nodes of the legal-move tree rooted at `pos`, `depth` plies deep.
///
/// `perft(pos, 0)` is `1` by definition. At depth 1 it returns the number of
/// legal moves without playing them.
pub fn perft(pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut list = MoveList::new();
    generate::<GEN_ALL>(pos, &mut list);
    if depth == 1 {
        return list.len() as u64;
    }
    let mut nodes = 0;
    for &mv in list.iter() {
        nodes += perft(&pos.make_move(mv), depth - 1);
    }
    nodes
}

/// Perft split by root move: for each legal move, the number of leaf nodes in
/// its subtree at `depth - 1` plies below it.
///
/// The result is sorted by the move's text, which makes it directly comparable
/// with the output of other engines' `divide` or `go perft`. The sum of the node
/// counts equals `perft(pos, depth)`.
pub fn divide(pos: &Position, depth: u32) -> Vec<(Move, u64)> {
    let mut list = MoveList::new();
    generate::<GEN_ALL>(pos, &mut list);
    let mut out: Vec<(Move, u64)> = list
        .iter()
        .map(|&mv| (mv, perft(&pos.make_move(mv), depth.saturating_sub(1))))
        .collect();
    out.sort_by_key(|(mv, _)| mv.to_string());
    out
}
