//! A fixed-depth search benchmark.
//!
//! [`run`] searches a fixed list of positions ([`BENCH_FENS`]) to a fixed depth
//! with a fresh transposition table and one thread, and reports the total number
//! of nodes and the elapsed time. Two properties make this useful:
//!
//! * The **node count is deterministic** for a given build and depth, because the
//!   search is single-threaded and depth-limited. Any change in the count after an
//!   edit means the *search behaviour* changed, which is exactly what a
//!   "functional" refactor must not do. Quote the node count in commit messages
//!   when a change is meant to be behaviour-neutral.
//! * The **nodes-per-second** figure measures raw speed: move generation,
//!   make-move, evaluation and table access.
//!
//! Run it from the command line with `wustite bench [depth]` (default depth 12),
//! or from a UCI session with `bench [depth]` (default 10).
//!
//! # Examples
//!
//! ```no_run
//! use wustite::bench;
//! use wustite::search::Params;
//!
//! let report = bench::run(10, Params::default(), false);
//! println!("{} nodes in {:.2}s", report.nodes, report.seconds);
//! ```

use std::time::Instant;

use crate::board::Position;
use crate::search::{Heuristics, Params, SearchLimits, SharedState, think};
use crate::tt::TranspositionTable;

/// The benchmark positions: the starting position, a dozen tactical and
/// positional middlegames, and two endings, so that the benchmark exercises both
/// the busy and the quiet parts of the search.
pub const BENCH_FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    "rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14",
    "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14",
    "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
    "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
    "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
    "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
    "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
    "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
    "8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
];

/// The outcome of a [`run`].
pub struct BenchReport {
    /// Total nodes searched over all positions.
    pub nodes: u64,
    /// Wall-clock time in seconds.
    pub seconds: f64,
}

/// Searches every position in [`BENCH_FENS`] to `depth` plies and returns the
/// totals.
///
/// `params` selects which search features are enabled; pass
/// [`Params::default`](crate::search::Params) for the shipping configuration.
/// When `verbose` is set, one line per position (best move and node count) is
/// printed. Each position is searched with an emptied transposition table and
/// emptied history tables so that results do not depend on order.
pub fn run(depth: i32, params: Params, verbose: bool) -> BenchReport {
    let tt = TranspositionTable::new(16);
    let mut heuristics = vec![Heuristics::new()];
    let start = Instant::now();
    let mut total = 0u64;

    for (i, fen) in BENCH_FENS.iter().enumerate() {
        let Ok(pos) = Position::from_fen(fen) else {
            println!("bench position {} is not valid: {fen}", i + 1);
            continue;
        };
        tt.clear();
        heuristics[0].clear();
        let shared = SharedState::new();
        let limits = SearchLimits::depth_only(depth);
        let result = think(
            &pos,
            &[pos.hash()],
            &limits,
            &tt,
            &shared,
            params,
            &mut heuristics,
            false,
        );
        total += result.nodes;
        if verbose {
            println!(
                "position {:>2}/{}: bestmove {} nodes {}",
                i + 1,
                BENCH_FENS.len(),
                result.best_move,
                result.nodes
            );
        }
    }

    BenchReport {
        nodes: total,
        seconds: start.elapsed().as_secs_f64(),
    }
}
