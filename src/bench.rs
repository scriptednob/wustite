//! Fixed-depth search benchmark.
//!
//! [`run`] searches [`BENCH_FENS`] to a fixed depth on one thread with a clean
//! hash table and reports total nodes and wall time.
//!
//! Because it's single-threaded and depth-limited, the total node count is
//! completely deterministic for a given build. If a refactor isn't supposed to
//! change search behavior and the bench node count shifts, something broke. It's
//! also a quick way to check raw NPS.
//!
//! Run from the CLI with `wustite bench [depth]` (default 12), or inside UCI
//! with `bench [depth]` (default 10).
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

/// Mix of opening, middlegame, and endgame FENs searched by [`run`].
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

/// Totals returned by [`run`].
pub struct BenchReport {
    /// Sum of searched nodes across all benchmark positions.
    pub nodes: u64,
    /// Elapsed wall-clock time in seconds.
    pub seconds: f64,
}

/// Search every position in [`BENCH_FENS`] to `depth` plies and return the totals.
///
/// Clears the transposition table and history tables before each position so
/// earlier positions don't affect later ones. Pass `verbose = true` to print
/// each position's best move and node count as it finishes.
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
