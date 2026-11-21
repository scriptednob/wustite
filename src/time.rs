//! UCI `go` parameter parsing and time management.
//!
//! [`GoParams::parse`] pulls clock and limit fields out of a `go` command, and
//! [`build_limits`] turns them into the [`SearchLimits`] used by the search.
//!
//! # Soft and hard limits
//!
//! Clock-based searches get two deadlines:
//!
//! * The soft limit is checked after each completed iteration. Once it passes,
//!   we stop rather than starting a deeper depth we probably won't finish.
//! * The hard limit is checked every 1024 nodes inside the search. If it trips,
//!   the search aborts right away and uses the last completed depth.
//!
//! # Clock formula
//!
//! After subtracting `Move Overhead` from the remaining clock (`available`):
//!
//! ```text
//! moves_to_go = movestogo (default 30), clamped to 1..=50
//! optimum     = min(available / moves_to_go + 3/4 * increment, 3/4 * available)
//! hard        = min(4 * optimum, 7/10 * available)
//! soft        = 6/10 * optimum, clamped to 1..=hard
//! ```
//!
//! Letting `hard` reach up to 4x the optimum gives an unfinished iteration room
//! to recover from a fail-low while capping any single move at 70% of the clock.
//! `movetime` sets both limits to `movetime - overhead`.
//!
//! ```
//! use wustite::time::GoParams;
//!
//! let go = GoParams::parse(&["wtime", "60000", "btime", "55000", "winc", "1000", "depth", "20"]);
//! assert_eq!(go.wtime, Some(60_000));
//! assert_eq!(go.btime, Some(55_000));
//! assert_eq!(go.winc, Some(1_000));
//! assert_eq!(go.depth, Some(20));
//! assert!(!go.infinite);
//! ```

use std::time::{Duration, Instant};

use crate::search::{MAX_PLY, SearchLimits};
use crate::types::Color;

/// Parsed arguments from a UCI `go` command. Unspecified fields stay `None` or `false`.
#[derive(Clone, Copy, Debug, Default)]
pub struct GoParams {
    /// White's remaining clock in milliseconds.
    pub wtime: Option<u64>,
    /// Black's remaining clock in milliseconds.
    pub btime: Option<u64>,
    /// White's increment per move in milliseconds.
    pub winc: Option<u64>,
    /// Black's increment per move in milliseconds.
    pub binc: Option<u64>,
    /// Moves left until the next time control (clamped to at least 1).
    pub movestogo: Option<u64>,
    /// Exact search time in milliseconds.
    pub movetime: Option<u64>,
    /// Depth limit in plies, clamped to `1..MAX_PLY - 1`.
    pub depth: Option<i32>,
    /// Approximate node budget.
    pub nodes: Option<u64>,
    /// Keep searching until `stop` arrives.
    pub infinite: bool,
    /// Search on the opponent's clock until `ponderhit` or `stop`.
    pub ponder: bool,
    /// Run a perft divide to this depth instead of searching.
    pub perft: Option<u32>,
}

impl GoParams {
    /// Parse the tokens after `go`.
    ///
    /// Skips unknown keywords and unparseable numbers so non-standard GUI extensions
    /// don't break anything. Negative clock values are clamped to 0.
    pub fn parse(tokens: &[&str]) -> GoParams {
        let mut go = GoParams::default();
        let mut i = 0;
        while i < tokens.len() {
            let value = tokens.get(i + 1).and_then(|t| t.parse::<i64>().ok());
            let mut consumed = 2;
            match (tokens[i], value) {
                ("wtime", Some(v)) => go.wtime = Some(v.max(0) as u64),
                ("btime", Some(v)) => go.btime = Some(v.max(0) as u64),
                ("winc", Some(v)) => go.winc = Some(v.max(0) as u64),
                ("binc", Some(v)) => go.binc = Some(v.max(0) as u64),
                ("movestogo", Some(v)) => go.movestogo = Some(v.max(1) as u64),
                ("movetime", Some(v)) => go.movetime = Some(v.max(1) as u64),
                ("depth", Some(v)) => go.depth = Some(v.clamp(1, MAX_PLY as i64 - 1) as i32),
                ("nodes", Some(v)) => go.nodes = Some(v.max(1) as u64),
                ("perft", Some(v)) => go.perft = Some(v.max(1) as u32),
                ("infinite", _) => {
                    go.infinite = true;
                    consumed = 1;
                }
                ("ponder", _) => {
                    go.ponder = true;
                    consumed = 1;
                }
                _ => consumed = 1,
            }
            i += consumed;
        }
        go
    }
}

/// Turn parsed `go` options into [`SearchLimits`] for `side`.
///
/// `overhead_ms` is subtracted from clock/movetime budgets to absorb GUI lag,
/// and `start` anchors all elapsed-time checks.
///
/// ```
/// use std::time::{Duration, Instant};
/// use wustite::time::{build_limits, GoParams};
/// use wustite::Color;
///
/// let go = GoParams::parse(&["movetime", "1000"]);
/// let limits = build_limits(&go, Color::White, 30, Instant::now());
/// assert_eq!(limits.soft, Some(Duration::from_millis(970)));
/// assert_eq!(limits.hard, Some(Duration::from_millis(970)));
///
/// let go = GoParams::parse(&["infinite"]);
/// let limits = build_limits(&go, Color::White, 30, Instant::now());
/// assert!(limits.infinite && limits.soft.is_none());
/// ```
pub fn build_limits(go: &GoParams, side: Color, overhead_ms: u64, start: Instant) -> SearchLimits {
    let mut limits = SearchLimits {
        depth: go.depth.unwrap_or(MAX_PLY as i32 - 1),
        nodes: go.nodes.unwrap_or(0),
        soft: None,
        hard: None,
        infinite: go.infinite,
        start,
    };
    if go.infinite {
        return limits;
    }

    let (time, increment) = if side == Color::White {
        (go.wtime, go.winc.unwrap_or(0))
    } else {
        (go.btime, go.binc.unwrap_or(0))
    };

    if let Some(movetime) = go.movetime {
        let budget = movetime.saturating_sub(overhead_ms).max(1);
        limits.soft = Some(Duration::from_millis(budget));
        limits.hard = Some(Duration::from_millis(budget));
    } else if let Some(time) = time {
        let available = time.saturating_sub(overhead_ms).max(1);
        let moves_to_go = go.movestogo.unwrap_or(30).clamp(1, 50);
        let optimum = (available / moves_to_go + increment * 3 / 4)
            .min(available * 3 / 4)
            .max(1);
        let hard = (optimum * 4).min(available * 7 / 10).max(1);
        let soft = (optimum * 6 / 10).clamp(1, hard);
        limits.soft = Some(Duration::from_millis(soft));
        limits.hard = Some(Duration::from_millis(hard));
    }
    limits
}
