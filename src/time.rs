//! Parsing of the UCI `go` command and time management.
//!
//! [`GoParams`] holds the raw parameters of a `go` command. [`build_limits`] turns
//! them, together with the side to move, into the [`SearchLimits`]
//! the search obeys.
//!
//! # Two limits: soft and hard
//!
//! Time-controlled searches get two deadlines:
//!
//! * The **soft limit** is checked *between* iterative-deepening iterations. If
//!   it has passed, the search stops without starting another, deeper iteration
//!   that could not finish.
//! * The **hard limit** is checked *inside* the search every 1024 nodes. If it
//!   passes, the search aborts immediately and the result of the last completed
//!   iteration is used.
//!
//! # Budget formula
//!
//! For a clock-based search (`wtime`/`btime`), with `available` equal to the
//! remaining time minus the *Move Overhead* option:
//!
//! ```text
//! moves_to_go = movestogo, or 30 when not given, clamped to 1..=50
//! optimum     = min(available / moves_to_go + 3/4 * increment, 3/4 * available)
//! hard        = min(4 * optimum, 7/10 * available)
//! soft        = 6/10 * optimum, clamped to 1..=hard
//! ```
//!
//! The hard limit allows an unfinished iteration to run about four times past the
//! optimum if that is what it takes to resolve a fail-low or a new best move,
//! while never spending more than 70% of the remaining clock on a single move.
//!
//! `movetime` searches use `movetime - overhead` for both limits. `depth`,
//! `nodes` and `infinite` set fixed limits instead (no clock limits for
//! `infinite`). Every limit that does not apply is simply absent.
//!
//! # Examples
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

/// The parameters of a UCI `go` command. A field is `None` (or `false`) when the
/// GUI did not send it.
#[derive(Clone, Copy, Debug, Default)]
pub struct GoParams {
    /// White's remaining time in milliseconds.
    pub wtime: Option<u64>,
    /// Black's remaining time in milliseconds.
    pub btime: Option<u64>,
    /// White's increment per move in milliseconds.
    pub winc: Option<u64>,
    /// Black's increment per move in milliseconds.
    pub binc: Option<u64>,
    /// Moves remaining until the next time control. Values below 1 are raised to 1.
    pub movestogo: Option<u64>,
    /// Search for exactly this many milliseconds.
    pub movetime: Option<u64>,
    /// Search to this depth. Clamped to `1..MAX_PLY - 1`.
    pub depth: Option<i32>,
    /// Stop after about this many nodes.
    pub nodes: Option<u64>,
    /// Search until `stop` is received.
    pub infinite: bool,
    /// Search on the opponent's time. The search runs unlimited until `ponderhit` or `stop`.
    pub ponder: bool,
    /// Not a search: count leaf nodes to this depth, like `go perft N` in Stockfish.
    pub perft: Option<u32>,
}

impl GoParams {
    /// Parses the tokens that follow `go`.
    ///
    /// Unknown tokens are skipped and numeric values that fail to parse are ignored,
    /// so a GUI sending extra parameters cannot break the command. Negative times
    /// are treated as zero.
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

/// Computes the [`SearchLimits`] for a `go` command.
///
/// `side` selects which clock applies, `overhead_ms` is the *Move Overhead*
/// reserved for communication delays, and `start` is the moment the search
/// began, which the search measures all elapsed time against. See the
/// [module documentation](crate::time) for the formula.
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
