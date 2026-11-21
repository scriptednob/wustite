//! Alpha-beta search (fail-soft PVS), iterative deepening, pruning, and Lazy SMP.
//!
//! Use [`think`] when you want full control over limits, shared stop flags,
//! persistent heuristics, and multi-threading (what [`crate::uci`] calls), or
//! [`search_depth`] for quick single-threaded fixed-depth searches.
//!
//! Scores are negamax centipawns inside `(-INFINITY, INFINITY)`. Forced mate in
//! `N` plies scores `MATE - N`, so shorter mates always beat longer ones;
//! anything at or past [`MATE_IN_MAX`] formats as `mate` via [`format_score`].
//!
//! # Search structure
//!
//! * **Root loop:** Iterative deepening from depth 1 up to the limit, adding
//!   aspiration windows from depth 5 onward (`aspiration_delta`, growing by 50%
//!   on fail-high/low and opening fully past 500 cp).
//! * **PVS core:** Fail-soft negamax generic over `const PV: bool`. The first
//!   move at a node searches the full window; later moves try a zero-window
//!   scout `(alpha, alpha + 1)` first and only re-search on a surprise.
//! * **Quiescence:** At depth 0, resolves captures and promotions with
//!   stand-pat and SEE filtering (or all evasions when in check).
//! * **Lazy SMP:** With multiple threads, helpers search the same root using
//!   staggered depth skips (`SKIP_SIZE`/`SKIP_PHASE`) and cooperate strictly
//!   through the shared transposition table. Only thread 0 checks limits and
//!   prints UCI `info` lines.
//!
//! # Techniques and flags
//!
//! Each toggle below lives in [`Params`] and can be flipped via hidden UCI
//! options for SPRT testing. Flags that are off by default didn't pass testing
//! yet but remain in the codebase to experiment with. All pruning/reductions
//! back off near mate scores so forced mates stay exact.
//!
//! | Technique | Default | Summary |
//! | --- | --- | --- |
//! | Check extension | always | Extend 1 ply while in check |
//! | Mate-distance pruning | always | Tighten `(alpha, beta)` to fastest possible mate from this ply |
//! | TT cutoffs | always | Return deep enough matching bounds on non-PV nodes |
//! | Internal iterative reduction (`iir`) | on | Reduce 1 ply at depth >= 4 when no TT move exists |
//! | Reverse futility (`rfp`) | on | At depth <= 8, return eval if `eval - margin * (depth - improving) >= beta` |
//! | Null-move pruning (`nmp`) | on | Free move with `R = 3 + depth/4 + min((eval - beta)/200, 3)`; verified at depth >= 12 |
//! | Late move reductions (`lmr`) | on | Reduce late quiet moves via compile-time log table; re-search if they beat alpha |
//! | Late move pruning (`lmp`) | on | At depth <= 6, skip late quiets past `3 + depth^2` (halved if not improving) |
//! | Futility pruning (`futility`) | on | At depth <= 6, skip quiets when `eval + 100 + 120 * depth <= alpha` |
//! | SEE pruning (`seepruning`) | on | At depth <= 8, skip moves whose SEE falls below `-margin * depth` |
//! | Singular extensions (`singular`) | on | Extend a clearly superior TT move (+1, or +2 non-PV), multi-cut, or negative-extend |
//! | TT eval (`tteval`) | on | Refine static eval with a directionally valid TT score |
//! | Continuation history (`conthistory`) | on | Quiet history conditioned on 1- and 2-ply previous moves |
//! | Razoring (`razoring`) | off | Verify low-eval shallow nodes with qsearch |
//! | ProbCut (`probcut`) | off | Shallow tactical search against a raised beta |
//! | Capture history (`capturehistory`) | off | History table for captures by `(piece, to, victim)` |
//! | History pruning / LMR (`histpruning`, `histlmr`) | off | Prune or adjust LMR based on quiet history |
//!
//! # Move ordering
//!
//! Moves are scored once and picked lazily via partial selection sort (since
//! cutoffs usually happen within the first few moves):
//! 1. TT move
//! 2. Good/equal captures (SEE >= 0) and queen promotions (MVV-LVA)
//! 3. Killer 1, Killer 2, Counter move
//! 4. Quiet moves by butterfly + continuation history (updated with gravity)
//! 5. Losing captures and under-promotions
//!
//! ```
//! use wustite::Position;
//! use wustite::search::{search_depth, MATE_IN_MAX};
//!
//! let pos: Position = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1".parse().unwrap();
//! let result = search_depth(&pos, 4, 4);
//! assert_eq!(result.best_move.to_string(), "a1a8");
//! assert!(result.score >= MATE_IN_MAX);
//! ```
//!
//! Full call with explicit limits and state:
//!
//! ```
//! use wustite::search::{think, Heuristics, Params, SearchLimits, SharedState};
//! use wustite::tt::TranspositionTable;
//! use wustite::Position;
//!
//! let pos = Position::startpos();
//! let tt = TranspositionTable::new(16);
//! let shared = SharedState::new();
//! let limits = SearchLimits::depth_only(6);
//! let mut heuristics = vec![Heuristics::new()];
//!
//! let result = think(
//!     &pos,
//!     &[pos.hash()],
//!     &limits,
//!     &tt,
//!     &shared,
//!     Params::default(),
//!     &mut heuristics,
//!     false,
//! );
//! assert!(!result.best_move.is_none());
//! assert_eq!(result.depth, 6);
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::board::Position;
use crate::eval::{bare_king_eval, evaluate as classical_evaluate};
use crate::movegen::{GEN_ALL, GEN_TACTICAL, MAX_MOVES, MoveList, generate, legal_moves};
use crate::nnue::{self, Accumulator, Network};
use crate::see::{SEE_VALUE, see_ge};
use crate::tt::{Bound, TranspositionTable};
use crate::types::{Color, Move, PieceType};

/// Maximum search ply (including quiescence), used to size stack arrays.
pub const MAX_PLY: usize = 128;
/// Score bound larger than any real evaluation or mate score.
pub const INFINITY: i32 = 32_000;
/// Base checkmate score (`MATE - ply` for winning mates).
pub const MATE: i32 = 31_000;
/// Threshold above which a score represents a forced mate.
pub const MATE_IN_MAX: i32 = MATE - MAX_PLY as i32;

/// Sentinel for missing static eval (e.g. when in check).
const NO_EVAL: i32 = 32_001;
/// Capacity of the per-thread repetition hash stack.
const HIST_CAP: usize = 512;
/// Gravity saturation limit for history tables.
const HISTORY_MAX: i32 = 16_384;
/// Stack size given to Lazy SMP helper threads.
const THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Move-ordering score for the TT move.
const TT_MOVE_SCORE: i32 = 1 << 30;
/// Base move-ordering score for SEE >= 0 captures and queen promotions.
const GOOD_CAPTURE: i32 = 1 << 28;
/// Base move-ordering score for killer moves.
const KILLER_SCORE: i32 = 1 << 26;
/// Move-ordering score for the counter move.
const COUNTER_SCORE: i32 = 1 << 25;
/// Base move-ordering score for losing captures and under-promotions.
const BAD_CAPTURE: i32 = -(1 << 24);

/// Lazy SMP helper depth-skip block sizes.
const SKIP_SIZE: [i32; 20] = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4];
/// Lazy SMP helper depth-skip phase offsets.
const SKIP_PHASE: [i32; 20] = [0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7];

/// Const-eval natural log via range reduction + atanh series (used to build [`LMR_TABLE`] at compile time).
const fn ln(x: f64) -> f64 {
    let mut m = x;
    let mut k = 0i32;
    while m >= 2.0 {
        m /= 2.0;
        k += 1;
    }
    while m < 1.0 {
        m *= 2.0;
        k -= 1;
    }
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let mut term = y;
    let mut sum = 0.0;
    let mut n = 1;
    while n < 40 {
        sum += term / n as f64;
        term *= y2;
        n += 2;
    }
    k as f64 * core::f64::consts::LN_2 + 2.0 * sum
}

/// Build the compile-time LMR table (`0.75 + ln(depth) * ln(move_count) / 2.25`).
const fn build_lmr() -> [[u8; 64]; 64] {
    let mut table = [[0u8; 64]; 64];
    let mut depth = 1;
    while depth < 64 {
        let mut count = 1;
        while count < 64 {
            let value = 0.75 + ln(depth as f64) * ln(count as f64) / 2.25;
            table[depth][count] = value as u8;
            count += 1;
        }
        depth += 1;
    }
    table
}

/// Precomputed LMR reductions indexed `[depth][move_count]`.
static LMR_TABLE: [[u8; 64]; 64] = build_lmr();

/// Feature switches and pruning margins for the search.
///
/// [`Params::default`] is the shipping setup; [`Params::baseline`] turns every
/// search enhancement off for A/B testing. Individual fields can be tweaked by
/// name via [`Params::set`] (and UCI `setoption`).
///
/// ```
/// use wustite::search::Params;
///
/// let mut params = Params::default();
/// assert!(params.nmp);
/// assert!(params.set("NMP", 0));
/// assert!(!params.nmp);
/// assert!(params.set("rfpmargin", 90));
/// assert_eq!(params.rfp_margin, 90);
/// assert!(!params.set("not-an-option", 1));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Enable root aspiration windows from depth 5.
    pub aspiration: bool,
    /// Enable late move reductions.
    pub lmr: bool,
    /// Enable null-move pruning.
    pub nmp: bool,
    /// Enable reverse futility pruning.
    pub rfp: bool,
    /// Enable shallow razoring into quiescence search.
    pub razoring: bool,
    /// Enable quiet-move futility pruning.
    pub futility: bool,
    /// Enable late move pruning for quiet moves.
    pub lmp: bool,
    /// Enable SEE pruning at shallow depths.
    pub see_pruning: bool,
    /// Enable internal iterative reduction when no TT move is found.
    pub iir: bool,
    /// Enable capture history in move ordering.
    pub capture_history: bool,
    /// Enable 1- and 2-ply continuation history.
    pub cont_history: bool,
    /// Enable singular extensions (plus double ext, multi-cut, and negative ext).
    pub singular: bool,
    /// Enable ProbCut.
    pub probcut: bool,
    /// Refine static eval using directionally valid TT scores.
    pub tt_eval: bool,
    /// Scale LMR reductions by quiet move history.
    pub hist_lmr: bool,
    /// Use NNUE evaluation (`false` falls back to classical PeSTO).
    pub nnue: bool,
    /// Prune quiet moves with strongly negative history at low depth.
    pub hist_pruning: bool,
    /// History pruning threshold (`history < -margin * depth`).
    pub hist_prune_margin: i32,
    /// Reverse futility margin per ply (cp).
    pub rfp_margin: i32,
    /// Razoring margin per ply (cp).
    pub razor_margin: i32,
    /// Base constant in the futility pruning margin (cp).
    pub futility_base: i32,
    /// Per-ply slope in the futility pruning margin (cp).
    pub futility_slope: i32,
    /// Base ply reduction for null-move pruning.
    pub nmp_base: i32,
    /// Minimum depth at which null-move cutoffs run a verification search.
    pub nmp_verify_depth: i32,
    /// SEE pruning margin per ply for captures (`-margin * depth`).
    pub see_capture_margin: i32,
    /// SEE pruning margin per ply for quiet moves (`-margin * depth`).
    pub see_quiet_margin: i32,
    /// Starting half-width for aspiration windows (cp).
    pub aspiration_delta: i32,
    /// Centipawn offset above beta tested by ProbCut.
    pub probcut_margin: i32,
    /// Minimum depth required to test singular extensions.
    pub singular_depth: i32,
}

impl Params {
    /// Plain alpha-beta baseline: all pruning/reduction flags off, NNUE on, default margins intact.
    ///
    /// ```
    /// use wustite::search::Params;
    ///
    /// let base = Params::baseline();
    /// assert!(!base.lmr && !base.nmp && !base.rfp);
    /// assert!(base.nnue);
    /// ```
    pub const fn baseline() -> Params {
        Params {
            aspiration: false,
            lmr: false,
            nmp: false,
            rfp: false,
            razoring: false,
            futility: false,
            lmp: false,
            see_pruning: false,
            iir: false,
            capture_history: false,
            cont_history: false,
            singular: false,
            probcut: false,
            tt_eval: false,
            hist_lmr: false,
            nnue: true,
            hist_pruning: false,
            hist_prune_margin: 3000,
            rfp_margin: 80,
            razor_margin: 250,
            futility_base: 100,
            futility_slope: 120,
            nmp_base: 3,
            nmp_verify_depth: 12,
            see_capture_margin: 100,
            see_quiet_margin: 60,
            aspiration_delta: 20,
            probcut_margin: 200,
            singular_depth: 8,
        }
    }

    /// Update a field by case-insensitive name (no spaces). Returns `false` if unknown.
    ///
    /// Booleans treat `0` as `false` and anything else as `true`.
    ///
    /// | Booleans | Numbers |
    /// | --- | --- |
    /// | `aspiration`, `lmr`, `nmp`, `rfp`, `razoring`, `futility`, `lmp`, `seepruning`, `iir`, `capturehistory`, `conthistory`, `singular`, `probcut`, `tteval`, `histlmr`, `usennue`, `histpruning` | `rfpmargin`, `razormargin`, `futilitybase`, `futilityslope`, `nmpbase`, `nmpverifydepth`, `seecapturemargin`, `seequietmargin`, `aspirationdelta`, `probcutmargin`, `singulardepth`, `histprunemargin` |
    pub fn set(&mut self, name: &str, value: i32) -> bool {
        let flag = value != 0;
        match name.to_ascii_lowercase().as_str() {
            "aspiration" => self.aspiration = flag,
            "lmr" => self.lmr = flag,
            "nmp" => self.nmp = flag,
            "rfp" => self.rfp = flag,
            "razoring" => self.razoring = flag,
            "futility" => self.futility = flag,
            "lmp" => self.lmp = flag,
            "seepruning" => self.see_pruning = flag,
            "iir" => self.iir = flag,
            "capturehistory" => self.capture_history = flag,
            "conthistory" => self.cont_history = flag,
            "singular" => self.singular = flag,
            "probcut" => self.probcut = flag,
            "tteval" => self.tt_eval = flag,
            "histlmr" => self.hist_lmr = flag,
            "usennue" => self.nnue = flag,
            "histpruning" => self.hist_pruning = flag,
            "rfpmargin" => self.rfp_margin = value,
            "razormargin" => self.razor_margin = value,
            "futilitybase" => self.futility_base = value,
            "futilityslope" => self.futility_slope = value,
            "nmpbase" => self.nmp_base = value,
            "nmpverifydepth" => self.nmp_verify_depth = value,
            "seecapturemargin" => self.see_capture_margin = value,
            "seequietmargin" => self.see_quiet_margin = value,
            "aspirationdelta" => self.aspiration_delta = value,
            "probcutmargin" => self.probcut_margin = value,
            "singulardepth" => self.singular_depth = value,
            "histprunemargin" => self.hist_prune_margin = value,
            _ => return false,
        }
        true
    }
}

impl Default for Params {
    fn default() -> Params {
        Params {
            aspiration: true,
            lmr: true,
            nmp: true,
            rfp: true,
            futility: true,
            lmp: true,
            see_pruning: true,
            tt_eval: true,
            cont_history: true,
            singular: true,
            iir: true,
            ..Params::baseline()
        }
    }
}

/// Depth, node, and clock limits controlling when [`think`] stops.
#[derive(Clone, Copy, Debug)]
pub struct SearchLimits {
    /// Highest completed iteration depth.
    pub depth: i32,
    /// Total nodes searched across all threads.
    pub nodes: u64,
    /// Soft deadline checked between iterations.
    pub soft: Option<Duration>,
    /// Hard deadline checked inside the search every 1024 nodes.
    pub hard: Option<Duration>,
    /// Ignore time limits and keep going until `shared.stop` is raised.
    pub infinite: bool,
    /// Timestamp when the search started.
    pub start: Instant,
}

impl SearchLimits {
    /// Convenience limits with only a depth cap and no time/node limits.
    pub fn depth_only(depth: i32) -> SearchLimits {
        SearchLimits {
            depth,
            nodes: 0,
            soft: None,
            hard: None,
            infinite: false,
            start: Instant::now(),
        }
    }
}

/// Atomic flags and node counter shared between the UCI loop and all search threads.
pub struct SharedState {
    /// Raised to abort all running search threads.
    pub stop: AtomicBool,
    /// Combined node count flushed by searchers every 1024 nodes.
    pub nodes: AtomicU64,
    /// While `true`, clock limits are paused until `ponderhit` clears it.
    pub pondering: AtomicBool,
    /// Milliseconds already elapsed when `ponderhit` arrived.
    pub ponder_offset_ms: AtomicU64,
}

impl SharedState {
    /// Create a clean `SharedState` (not stopped, 0 nodes, not pondering).
    pub fn new() -> SharedState {
        SharedState {
            stop: AtomicBool::new(false),
            nodes: AtomicU64::new(0),
            pondering: AtomicBool::new(false),
            ponder_offset_ms: AtomicU64::new(0),
        }
    }
}

impl Default for SharedState {
    fn default() -> SharedState {
        SharedState::new()
    }
}

/// Flattened continuation-history size (`12 * 64 * 12 * 64`).
const CONT_SIZE: usize = 12 * 64 * 12 * 64;

/// Per-thread move-ordering tables (butterfly, counter-move, capture, and continuation history) persisted across moves in a game.
pub struct Heuristics {
    butterfly: [[[i16; 64]; 64]; 2],
    counter: [[Move; 64]; 12],
    capture: [[[i16; 6]; 64]; 12],
    cont: Box<[i16]>,
}

impl Heuristics {
    /// Allocate a zeroed `Heuristics` bundle on the heap.
    pub fn new() -> Box<Heuristics> {
        Box::new(Heuristics {
            butterfly: [[[0; 64]; 64]; 2],
            counter: [[Move::NONE; 64]; 12],
            capture: [[[0; 6]; 64]; 12],
            cont: vec![0i16; CONT_SIZE].into_boxed_slice(),
        })
    }

    /// Zero every table (called on `ucinewgame`).
    pub fn clear(&mut self) {
        self.butterfly = [[[0; 64]; 64]; 2];
        self.counter = [[Move::NONE; 64]; 12];
        self.capture = [[[0; 6]; 64]; 12];
        self.cont.fill(0);
    }
}

/// Flatten `(prev_piece, prev_to, piece, to)` into a continuation-history index.
#[inline(always)]
fn cont_index(prev_piece: usize, prev_to: usize, piece: usize, to: usize) -> usize {
    ((prev_piece * 64 + prev_to) * 12 + piece) * 64 + to
}

/// Apply a history bonus with gravity decay (`entry += bonus - entry * |bonus| / HISTORY_MAX`).
#[inline(always)]
fn gravity(entry: &mut i16, bonus: i32) {
    let current = *entry as i32;
    *entry = (current + bonus - current * bonus.abs() / HISTORY_MAX) as i16;
}

/// Extract `(attacker_piece, to_sq, victim_type)` for a capture, or `None` for quiet moves.
fn capture_slot(pos: &Position, mv: Move) -> Option<(usize, usize, usize)> {
    if !mv.is_capture() {
        return None;
    }
    let victim = if mv.is_en_passant() {
        PieceType::Pawn.idx()
    } else {
        pos.piece_on(mv.to_sq())?.piece_type().idx()
    };
    let piece = pos.piece_on(mv.from_sq())?.idx();
    Some((piece, mv.to_sq().idx(), victim))
}

/// Best move, score, depth, node count, and PV returned by [`think`].
#[derive(Clone, Debug, Default)]
pub struct SearchResult {
    /// Chosen root move (or [`Move::NONE`] if checkmated/stalemated).
    pub best_move: Move,
    /// Second move of the PV for UCI pondering, or [`Move::NONE`].
    pub ponder_move: Move,
    /// Score from the last finished iteration (format with [`format_score`]).
    pub score: i32,
    /// Highest completed iteration depth.
    pub depth: i32,
    /// Total nodes searched across all threads.
    pub nodes: u64,
    /// Principal variation from the root.
    pub pv: Vec<Move>,
}

/// Per-ply scratchpad (static eval, killers, last move/piece, and singular-excluded move).
#[derive(Clone, Copy)]
struct StackEntry {
    static_eval: i32,
    killers: [Move; 2],
    mv: Move,
    piece: usize,
    excluded: Move,
}

/// Default empty [`StackEntry`].
const EMPTY_ENTRY: StackEntry = StackEntry {
    static_eval: NO_EVAL,
    killers: [Move::NONE; 2],
    mv: Move::NONE,
    piece: 0,
    excluded: Move::NONE,
};

/// Convert a root-relative mate score to node-relative distance before storing in the TT.
#[inline(always)]
fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_IN_MAX {
        score + ply as i32
    } else if score <= -MATE_IN_MAX {
        score - ply as i32
    } else {
        score
    }
}

/// Inverse of [`score_to_tt`]: re-base a TT mate score onto the current `ply`.
#[inline(always)]
fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_IN_MAX {
        score - ply as i32
    } else if score <= -MATE_IN_MAX {
        score + ply as i32
    } else {
        score
    }
}

/// Return `true` if a TT bound can trigger an immediate cutoff for `(alpha, beta)`.
#[inline(always)]
fn bound_allows(bound: Bound, score: i32, alpha: i32, beta: i32) -> bool {
    match bound {
        Bound::Exact => true,
        Bound::Lower => score >= beta,
        Bound::Upper => score <= alpha,
        Bound::None => false,
    }
}

/// Format a search score as UCI `cp <n>` or `mate <moves>` (negative if being mated).
///
/// ```
/// use wustite::search::{format_score, MATE};
///
/// assert_eq!(format_score(35), "cp 35");
/// assert_eq!(format_score(MATE - 1), "mate 1");
/// assert_eq!(format_score(MATE - 3), "mate 2");
/// assert_eq!(format_score(-(MATE - 2)), "mate -1");
/// ```
pub fn format_score(score: i32) -> String {
    if score >= MATE_IN_MAX {
        format!("mate {}", (MATE - score + 1) / 2)
    } else if score <= -MATE_IN_MAX {
        format!("mate -{}", (MATE + score) / 2)
    } else {
        format!("cp {score}")
    }
}

/// Score captures and promotions via MVV-LVA + capture history, split into good/bad bands by SEE.
fn tactical_score(pos: &Position, mv: Move, history: i32) -> i32 {
    let victim = if mv.is_en_passant() {
        SEE_VALUE[PieceType::Pawn.idx()]
    } else {
        pos.piece_on(mv.to_sq())
            .map_or(0, |p| SEE_VALUE[p.piece_type().idx()])
    };
    let attacker = pos
        .piece_on(mv.from_sq())
        .map_or(0, |p| p.piece_type().idx() as i32);
    let mvv_lva = victim * 16 - attacker + history / 8;
    match mv.promotion_piece() {
        Some(PieceType::Queen) => GOOD_CAPTURE + 20_000 + mvv_lva,
        Some(_) => BAD_CAPTURE + mvv_lva,
        None => {
            if see_ge(pos, mv, 0) {
                GOOD_CAPTURE + mvv_lva
            } else {
                BAD_CAPTURE + mvv_lva
            }
        }
    }
}

/// Per-thread search state holding the ply stack, triangular PV table, repetition hashes, and NNUE accumulators.
struct Searcher<'a> {
    id: usize,
    tt: &'a TranspositionTable,
    shared: &'a SharedState,
    limits: &'a SearchLimits,
    params: Params,
    heur: &'a mut Heuristics,
    nodes: u64,
    flushed: u64,
    seldepth: usize,
    stopped: bool,
    root_depth: i32,
    nmp_min_ply: usize,
    stack: [StackEntry; MAX_PLY + 4],
    pv: [[Move; MAX_PLY + 1]; MAX_PLY + 1],
    pv_len: [usize; MAX_PLY + 1],
    hist: [u64; HIST_CAP],
    hist_len: usize,
    net: Option<Arc<Network>>,
    acc: Vec<Accumulator>,
}

impl<'a> Searcher<'a> {
    /// Set up a searcher for thread `id` (`0` = main thread), seeding repetition history and the root NNUE accumulator.
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: usize,
        tt: &'a TranspositionTable,
        shared: &'a SharedState,
        limits: &'a SearchLimits,
        params: Params,
        heur: &'a mut Heuristics,
        root: &Position,
        game_hist: &[u64],
    ) -> Searcher<'a> {
        let keep = game_hist.len().min(HIST_CAP - MAX_PLY - 8);
        let mut hist = [0u64; HIST_CAP];
        hist[..keep].copy_from_slice(&game_hist[game_hist.len() - keep..]);
        let mut hist_len = keep;
        if hist_len == 0 || hist[hist_len - 1] != root.hash() {
            hist[hist_len] = root.hash();
            hist_len += 1;
        }
        let net = if params.nnue { nnue::current() } else { None };
        let mut acc = Vec::new();
        if let Some(network) = &net {
            acc = vec![Accumulator::default(); MAX_PLY + 4];
            nnue::refresh(network, &mut acc[0], root);
        }
        Searcher {
            id,
            tt,
            shared,
            limits,
            params,
            heur,
            nodes: 0,
            flushed: 0,
            seldepth: 0,
            stopped: false,
            root_depth: 0,
            nmp_min_ply: 0,
            stack: [EMPTY_ENTRY; MAX_PLY + 4],
            pv: [[Move::NONE; MAX_PLY + 1]; MAX_PLY + 1],
            pv_len: [0; MAX_PLY + 1],
            hist,
            hist_len,
            net,
            acc,
        }
    }

    /// Evaluate `pos` at `ply` (mop-up -> NNUE accumulator -> classical PeSTO). Debug builds cross-check the accumulator.
    #[inline(always)]
    fn evaluate(&self, pos: &Position, ply: usize) -> i32 {
        if let Some(score) = bare_king_eval(pos) {
            return score;
        }
        match &self.net {
            Some(net) => {
                #[cfg(debug_assertions)]
                {
                    let mut fresh = Accumulator::default();
                    nnue::scalar_refresh(net, &mut fresh, pos);
                    assert!(
                        fresh.v == self.acc[ply].v,
                        "accumulator out of sync at ply {ply}: {}",
                        pos.to_fen()
                    );
                }
                nnue::evaluate(net, &self.acc[ply], pos.side_to_move())
                    .clamp(-MATE_IN_MAX + 1, MATE_IN_MAX - 1)
            }
            None => classical_evaluate(pos),
        }
    }

    /// Update `acc[ply + 1]` for `mv`. Must run right before descending (after any singular verification search at `ply`).
    #[inline(always)]
    fn update_accumulator(&mut self, pos: &Position, mv: Move, ply: usize) {
        if let Some(net) = &self.net {
            let (head, tail) = self.acc.split_at_mut(ply + 1);
            nnue::update(net, &head[ply], &mut tail[0], pos, mv);
        }
    }

    /// Update the child NNUE accumulator and return `pos.make_move(mv)`.
    #[inline(always)]
    fn make_move(&mut self, pos: &Position, mv: Move, ply: usize) -> Position {
        self.update_accumulator(pos, mv, ply);
        pos.make_move(mv)
    }

    /// Copy the parent accumulator into `ply + 1` and return `pos.make_null_move()`.
    #[inline(always)]
    fn make_null_move(&mut self, pos: &Position, ply: usize) -> Position {
        if self.net.is_some() {
            self.acc[ply + 1] = self.acc[ply];
        }
        pos.make_null_move()
    }

    /// Push local un-flushed nodes to `shared.nodes`.
    fn flush_nodes(&mut self) {
        let delta = self.nodes - self.flushed;
        self.flushed = self.nodes;
        self.shared.nodes.fetch_add(delta, Ordering::Relaxed);
    }

    /// Mark this searcher stopped and raise the global `shared.stop` flag.
    fn trigger_stop(&mut self) {
        self.stopped = true;
        self.shared.stop.store(true, Ordering::Relaxed);
    }

    /// Periodic 1024-node check: flush nodes, read `shared.stop`, and enforce node/hard-time limits on thread 0.
    fn poll(&mut self) {
        let delta = self.nodes - self.flushed;
        self.flushed = self.nodes;
        let total = self.shared.nodes.fetch_add(delta, Ordering::Relaxed) + delta;
        if self.root_depth <= 1 {
            return;
        }
        if self.shared.stop.load(Ordering::Relaxed) {
            self.stopped = true;
            return;
        }
        if self.id == 0 {
            if self.limits.nodes > 0 && total >= self.limits.nodes {
                self.trigger_stop();
                return;
            }
            if let Some(hard) = self.limits.hard
                && self.limit_clock() >= hard
            {
                self.trigger_stop();
            }
        }
    }

    /// Check insufficient material, 50-move rule (excluding checkmate), and repetition (2-fold inside tree, 3-fold across root).
    fn is_draw(&self, pos: &Position, ply: usize) -> bool {
        if pos.insufficient_material() {
            return true;
        }
        let halfmove = pos.halfmove_clock() as usize;
        if halfmove >= 100 {
            return !pos.in_check() || !legal_moves(pos).is_empty();
        }
        let end = halfmove.min(self.hist_len - 1);
        let key = pos.hash();
        let mut seen = 0;
        let mut back = 4;
        while back <= end {
            if self.hist[self.hist_len - 1 - back] == key {
                if back <= ply {
                    return true;
                }
                seen += 1;
                if seen >= 2 {
                    return true;
                }
            }
            back += 2;
        }
        false
    }

    /// Push a position hash onto the repetition stack.
    #[inline(always)]
    fn push_hash(&mut self, key: u64) {
        self.hist[self.hist_len] = key;
        self.hist_len += 1;
    }

    /// Pop the latest hash from the repetition stack.
    #[inline(always)]
    fn pop_hash(&mut self) {
        self.hist_len -= 1;
    }

    /// Prepend `mv` onto the child's PV line in the triangular table.
    fn update_pv(&mut self, ply: usize, mv: Move) {
        self.pv[ply][0] = mv;
        let child_len = self.pv_len[ply + 1];
        for i in 0..child_len {
            self.pv[ply][i + 1] = self.pv[ply + 1][i];
        }
        self.pv_len[ply] = child_len + 1;
    }

    /// Apply a gravity-scaled butterfly history update.
    fn update_history(&mut self, color: Color, mv: Move, bonus: i32) {
        let entry = &mut self.heur.butterfly[color.idx()][mv.from_sq().idx()][mv.to_sq().idx()];
        gravity(entry, bonus);
    }

    /// Apply a gravity-scaled update to 1- and 2-ply continuation history.
    fn update_cont(&mut self, ply: usize, piece: usize, mv: Move, bonus: i32) {
        let to = mv.to_sq().idx();
        for back in [1usize, 2] {
            if ply >= back {
                let prev = self.stack[ply - back];
                if !prev.mv.is_none() {
                    let index = cont_index(prev.piece, prev.mv.to_sq().idx(), piece, to);
                    gravity(&mut self.heur.cont[index], bonus);
                }
            }
        }
    }

    /// Combined butterfly + (optional) continuation history score for a quiet move.
    fn quiet_history(&self, us: Color, piece: usize, mv: Move, ply: usize) -> i32 {
        let to = mv.to_sq().idx();
        let mut score = self.heur.butterfly[us.idx()][mv.from_sq().idx()][to] as i32;
        if self.params.cont_history {
            for back in [1usize, 2] {
                if ply >= back {
                    let prev = &self.stack[ply - back];
                    if !prev.mv.is_none() {
                        let index = cont_index(prev.piece, prev.mv.to_sq().idx(), piece, to);
                        score += self.heur.cont[index] as i32;
                    }
                }
            }
        }
        score
    }

    /// Capture history lookup (returns `0` when disabled).
    fn capture_history(&self, pos: &Position, mv: Move) -> i32 {
        if !self.params.capture_history {
            return 0;
        }
        match capture_slot(pos, mv) {
            Some((piece, to, victim)) => self.heur.capture[piece][to][victim] as i32,
            None => 0,
        }
    }

    /// Update one capture-history entry with gravity.
    fn reward_capture(&mut self, pos: &Position, mv: Move, bonus: i32) {
        if let Some((piece, to, victim)) = capture_slot(pos, mv) {
            gravity(&mut self.heur.capture[piece][to][victim], bonus);
        }
    }

    /// Fill `scores` for every move in `list` (TT -> tactical -> killers -> counter -> quiet history).
    fn score_moves(
        &self,
        pos: &Position,
        list: &MoveList,
        tt_move: Move,
        ply: usize,
        scores: &mut [i32; MAX_MOVES],
    ) {
        let us = pos.side_to_move();
        let killers = self.stack[ply].killers;
        let counter = if ply > 0 && !self.stack[ply - 1].mv.is_none() {
            let prev = &self.stack[ply - 1];
            self.heur.counter[prev.piece][prev.mv.to_sq().idx()]
        } else {
            Move::NONE
        };
        for (i, &mv) in list.iter().enumerate() {
            scores[i] = if mv == tt_move {
                TT_MOVE_SCORE
            } else if mv.is_capture() || mv.is_promotion() {
                tactical_score(pos, mv, self.capture_history(pos, mv))
            } else if mv == killers[0] {
                KILLER_SCORE + 1
            } else if mv == killers[1] {
                KILLER_SCORE
            } else if mv == counter {
                COUNTER_SCORE
            } else {
                let piece = pos.piece_on(mv.from_sq()).map_or(0, |p| p.idx());
                self.quiet_history(us, piece, mv, ply)
            };
        }
    }

    /// Core fail-soft PVS alpha-beta recursion (`PV = true` on principal-variation nodes).
    ///
    /// Applies check extensions, draw/mate-distance checks, TT probing, static eval
    /// and improving detection, whole-node pruning (razoring, RFP, NMP, ProbCut),
    /// move-loop pruning (SEE, LMP, futility, history), singular extensions, LMR,
    /// and history updates on cutoffs.
    fn search<const PV: bool>(
        &mut self,
        pos: &Position,
        mut alpha: i32,
        mut beta: i32,
        mut depth: i32,
        ply: usize,
    ) -> i32 {
        let in_check = pos.in_check();
        if in_check {
            depth += 1;
        }
        if depth <= 0 {
            return self.qsearch::<PV>(pos, alpha, beta, ply);
        }

        if PV {
            self.pv_len[ply] = 0;
        }
        self.nodes += 1;
        if self.nodes & 1023 == 0 {
            self.poll();
        }
        if self.stopped {
            return 0;
        }
        if ply > self.seldepth {
            self.seldepth = ply;
        }

        let is_root = ply == 0;
        if !is_root {
            if self.is_draw(pos, ply) {
                return 0;
            }
            if ply >= MAX_PLY - 1 {
                return if in_check { 0 } else { self.evaluate(pos, ply) };
            }
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        let p = self.params;
        let us = pos.side_to_move();
        let key = pos.hash();
        let excluded = self.stack[ply].excluded;
        let mut tt_move = Move::NONE;
        let mut tt_eval = NO_EVAL;
        let mut tt_score = 0;
        let mut tt_depth = -1;
        let mut tt_bound = Bound::None;
        if let Some(entry) = self.tt.probe(key) {
            tt_move = entry.mv;
            tt_eval = entry.eval;
            tt_score = score_from_tt(entry.score, ply);
            tt_depth = entry.depth;
            tt_bound = entry.bound;
            if !PV
                && excluded.is_none()
                && entry.depth >= depth
                && bound_allows(entry.bound, tt_score, alpha, beta)
                && pos.halfmove_clock() < 90
            {
                return tt_score;
            }
        }

        let static_eval = if in_check {
            NO_EVAL
        } else if tt_eval != NO_EVAL {
            tt_eval
        } else {
            self.evaluate(pos, ply)
        };
        self.stack[ply].static_eval = static_eval;
        self.stack[ply + 1].killers = [Move::NONE; 2];
        let improving = !in_check
            && ply >= 2
            && self.stack[ply - 2].static_eval != NO_EVAL
            && static_eval > self.stack[ply - 2].static_eval;

        let mut eval = static_eval;
        if p.tt_eval && !in_check && tt_score.abs() < MATE_IN_MAX {
            let usable = match tt_bound {
                Bound::Exact => true,
                Bound::Lower => tt_score > static_eval,
                Bound::Upper => tt_score < static_eval,
                Bound::None => false,
            };
            if usable {
                eval = tt_score;
            }
        }

        if p.iir && depth >= 4 && tt_move.is_none() {
            depth -= 1;
        }

        if !PV && !in_check {
            if p.razoring
                && excluded.is_none()
                && depth <= 3
                && alpha.abs() < MATE_IN_MAX
                && eval + p.razor_margin * depth <= alpha
            {
                let value = self.qsearch::<false>(pos, alpha, alpha + 1, ply);
                if self.stopped {
                    return 0;
                }
                if value <= alpha {
                    return value;
                }
            }

            if p.rfp
                && depth <= 8
                && eval - p.rfp_margin * (depth - improving as i32) >= beta
                && beta > -MATE_IN_MAX
                && eval < MATE_IN_MAX
            {
                return eval;
            }

            if p.nmp
                && excluded.is_none()
                && depth >= 3
                && eval >= beta
                && beta > -MATE_IN_MAX
                && ply >= self.nmp_min_ply
                && !self.stack[ply - 1].mv.is_none()
                && pos.has_non_pawn_material(us)
            {
                let reduction = p.nmp_base + depth / 4 + ((eval - beta) / 200).min(3);
                let child = self.make_null_move(pos, ply);
                self.stack[ply].mv = Move::NONE;
                self.stack[ply].piece = 0;
                self.push_hash(child.hash());
                let score =
                    -self.search::<false>(&child, -beta, -beta + 1, depth - 1 - reduction, ply + 1);
                self.pop_hash();
                if self.stopped {
                    return 0;
                }
                if score >= beta {
                    let score = if score >= MATE_IN_MAX { beta } else { score };
                    if depth < p.nmp_verify_depth {
                        return score;
                    }
                    let saved = self.nmp_min_ply;
                    self.nmp_min_ply = ply + 3 * (depth - reduction) as usize / 4;
                    let verify = self.search::<false>(pos, beta - 1, beta, depth - reduction, ply);
                    self.nmp_min_ply = saved;
                    if self.stopped {
                        return 0;
                    }
                    if verify >= beta {
                        return score;
                    }
                }
            }

            if p.probcut
                && excluded.is_none()
                && depth >= 5
                && beta.abs() < MATE_IN_MAX
                && !(tt_depth >= depth - 3
                    && matches!(tt_bound, Bound::Exact | Bound::Upper)
                    && tt_score < beta + p.probcut_margin)
            {
                let pc_beta = beta + p.probcut_margin;
                let mut caps = MoveList::new();
                generate::<GEN_TACTICAL>(pos, &mut caps);
                let mut cap_scores = [0i32; MAX_MOVES];
                self.score_moves(pos, &caps, tt_move, ply, &mut cap_scores);
                let cap_len = caps.len();
                for i in 0..cap_len {
                    let mut pick = i;
                    for j in i + 1..cap_len {
                        if cap_scores[j] > cap_scores[pick] {
                            pick = j;
                        }
                    }
                    caps.swap(i, pick);
                    cap_scores.swap(i, pick);
                    let mv = caps[i];
                    if !see_ge(pos, mv, pc_beta - eval) {
                        continue;
                    }
                    let child = self.make_move(pos, mv, ply);
                    self.tt.prefetch(child.hash());
                    let moved = pos.piece_on(mv.from_sq()).map_or(0, |piece| piece.idx());
                    self.stack[ply].mv = mv;
                    self.stack[ply].piece = moved;
                    self.push_hash(child.hash());
                    let mut value = -self.qsearch::<false>(&child, -pc_beta, -pc_beta + 1, ply + 1);
                    if value >= pc_beta {
                        value = -self.search::<false>(
                            &child,
                            -pc_beta,
                            -pc_beta + 1,
                            depth - 4,
                            ply + 1,
                        );
                    }
                    self.pop_hash();
                    if self.stopped {
                        return 0;
                    }
                    if value >= pc_beta {
                        let result = value - (pc_beta - beta);
                        self.tt.store(
                            key,
                            mv,
                            score_to_tt(result, ply),
                            static_eval,
                            depth - 3,
                            Bound::Lower,
                        );
                        return result;
                    }
                }
            }
        }

        let mut list = MoveList::new();
        generate::<GEN_ALL>(pos, &mut list);
        if list.is_empty() {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }
        let mut scores = [0i32; MAX_MOVES];
        self.score_moves(pos, &list, tt_move, ply, &mut scores);

        let original_alpha = alpha;
        let mut best = -INFINITY;
        let mut best_move = Move::NONE;
        let mut move_count = 0i32;
        let mut quiets = [Move::NONE; 64];
        let mut quiet_count = 0usize;
        let mut captures = [Move::NONE; 32];
        let mut capture_count = 0usize;
        let mut skip_quiets = false;
        let len = list.len();

        for i in 0..len {
            let mut pick = i;
            for j in i + 1..len {
                if scores[j] > scores[pick] {
                    pick = j;
                }
            }
            list.swap(i, pick);
            scores.swap(i, pick);
            let mv = list[i];
            let is_capture = mv.is_capture();
            let is_quiet = !is_capture && !mv.is_promotion();
            if mv == excluded {
                continue;
            }
            move_count += 1;

            let mate_window = alpha >= MATE_IN_MAX || beta <= -MATE_IN_MAX;
            let can_prune = !is_root && best > -MATE_IN_MAX && !mate_window;
            if can_prune && p.see_pruning && depth <= 8 {
                let margin = if is_capture {
                    -p.see_capture_margin * depth
                } else {
                    -p.see_quiet_margin * depth
                };
                if !see_ge(pos, mv, margin) {
                    continue;
                }
            }

            let child = pos.make_move(mv);
            self.tt.prefetch(child.hash());
            let gives_check = child.in_check();

            if can_prune && is_quiet && !in_check && !gives_check {
                if skip_quiets {
                    continue;
                }
                if p.lmp && depth <= 6 {
                    let base = 3 + depth * depth;
                    let limit = if improving { base } else { base / 2 };
                    if move_count > limit {
                        skip_quiets = true;
                        continue;
                    }
                }
                if p.futility
                    && depth <= 6
                    && alpha.abs() < MATE_IN_MAX
                    && eval + p.futility_base + p.futility_slope * depth <= alpha
                {
                    skip_quiets = true;
                    continue;
                }
                if p.hist_pruning && depth <= 4 {
                    let piece = pos.piece_on(mv.from_sq()).map_or(0, |piece| piece.idx());
                    if self.quiet_history(us, piece, mv, ply) < -p.hist_prune_margin * depth {
                        continue;
                    }
                }
            }

            let mut extension = 0i32;
            if p.singular
                && !is_root
                && mv == tt_move
                && excluded.is_none()
                && depth >= p.singular_depth
                && tt_depth >= depth - 3
                && matches!(tt_bound, Bound::Lower | Bound::Exact)
                && tt_score.abs() < MATE_IN_MAX
                && ply < 2 * self.root_depth.max(1) as usize
            {
                let singular_beta = tt_score - depth * 2;
                self.stack[ply].excluded = mv;
                let value = self.search::<false>(
                    pos,
                    singular_beta - 1,
                    singular_beta,
                    (depth - 1) / 2,
                    ply,
                );
                self.stack[ply].excluded = Move::NONE;
                if self.stopped {
                    return 0;
                }
                if value < singular_beta {
                    extension = 1;
                    if !PV && value < singular_beta - 25 {
                        extension = 2;
                    }
                } else if singular_beta >= beta {
                    return singular_beta;
                } else if tt_score >= beta {
                    extension = -1;
                }
            }

            self.update_accumulator(pos, mv, ply);
            let moved = pos.piece_on(mv.from_sq()).map_or(0, |piece| piece.idx());
            self.stack[ply].mv = mv;
            self.stack[ply].piece = moved;
            self.push_hash(child.hash());

            let new_depth = depth - 1 + extension;
            let mut score;
            if move_count == 1 {
                score = -self.search::<PV>(&child, -beta, -alpha, new_depth, ply + 1);
            } else {
                let mut reduction = 0i32;
                if p.lmr && depth >= 3 && move_count > 2 && is_quiet && !mate_window {
                    reduction =
                        LMR_TABLE[depth.min(63) as usize][move_count.min(63) as usize] as i32;
                    if PV {
                        reduction -= 1;
                    }
                    if !improving {
                        reduction += 1;
                    }
                    if gives_check {
                        reduction -= 1;
                    }
                    if p.hist_lmr {
                        reduction -= self.quiet_history(us, moved, mv, ply) / 6144;
                    }
                }
                let reduced = if reduction > 0 {
                    (new_depth - reduction).clamp(1, new_depth.max(1))
                } else {
                    new_depth
                };
                score = -self.search::<false>(&child, -alpha - 1, -alpha, reduced, ply + 1);
                if score > alpha && reduced < new_depth {
                    score = -self.search::<false>(&child, -alpha - 1, -alpha, new_depth, ply + 1);
                }
                if PV && score > alpha && score < beta {
                    score = -self.search::<true>(&child, -beta, -alpha, new_depth, ply + 1);
                }
            }
            self.pop_hash();
            if self.stopped {
                return 0;
            }

            if score > best {
                best = score;
                if score > alpha {
                    best_move = mv;
                    alpha = score;
                    if PV {
                        self.update_pv(ply, mv);
                    }
                    if alpha >= beta {
                        if is_quiet {
                            self.reward_quiet(pos, ply, mv, depth, &quiets[..quiet_count]);
                        }
                        if p.capture_history {
                            let winner = if is_capture { mv } else { Move::NONE };
                            self.reward_captures(pos, depth, winner, &captures[..capture_count]);
                        }
                        break;
                    }
                }
            }
            if is_quiet && quiet_count < quiets.len() {
                quiets[quiet_count] = mv;
                quiet_count += 1;
            } else if is_capture && capture_count < captures.len() {
                captures[capture_count] = mv;
                capture_count += 1;
            }
        }

        let bound = if best >= beta {
            Bound::Lower
        } else if best > original_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        if excluded.is_none() {
            self.tt.store(
                key,
                best_move,
                score_to_tt(best, ply),
                static_eval,
                depth,
                bound,
            );
        }
        best
    }

    /// Boost capture history for `winner` and penalize earlier failed captures.
    fn reward_captures(&mut self, pos: &Position, depth: i32, winner: Move, tried: &[Move]) {
        let bonus = (155 * depth - 80).clamp(0, 2_000);
        if !winner.is_none() {
            self.reward_capture(pos, winner, bonus);
        }
        for &capture in tried {
            if capture != winner {
                self.reward_capture(pos, capture, -bonus);
            }
        }
    }

    /// Update killers, counter move, butterfly history, and continuation history on a quiet beta cutoff.
    fn reward_quiet(&mut self, pos: &Position, ply: usize, mv: Move, depth: i32, tried: &[Move]) {
        let us = pos.side_to_move();
        if self.stack[ply].killers[0] != mv {
            self.stack[ply].killers[1] = self.stack[ply].killers[0];
            self.stack[ply].killers[0] = mv;
        }
        let bonus = (155 * depth - 80).clamp(0, 2_000);
        self.update_history(us, mv, bonus);
        for &quiet in tried {
            self.update_history(us, quiet, -bonus);
        }
        if self.params.cont_history {
            let piece = pos.piece_on(mv.from_sq()).map_or(0, |p| p.idx());
            self.update_cont(ply, piece, mv, bonus);
            for &quiet in tried {
                let piece = pos.piece_on(quiet.from_sq()).map_or(0, |p| p.idx());
                self.update_cont(ply, piece, quiet, -bonus);
            }
        }
        if ply > 0 && !self.stack[ply - 1].mv.is_none() {
            let prev = self.stack[ply - 1];
            self.heur.counter[prev.piece][prev.mv.to_sq().idx()] = mv;
        }
    }

    /// Quiescence search at depth 0: stand-pat + SEE-good captures/promotions, or full evasions when in check.
    fn qsearch<const PV: bool>(
        &mut self,
        pos: &Position,
        mut alpha: i32,
        beta: i32,
        ply: usize,
    ) -> i32 {
        if PV {
            self.pv_len[ply] = 0;
        }
        self.nodes += 1;
        if self.nodes & 1023 == 0 {
            self.poll();
        }
        if self.stopped {
            return 0;
        }
        if ply > self.seldepth {
            self.seldepth = ply;
        }
        if self.is_draw(pos, ply) {
            return 0;
        }
        let in_check = pos.in_check();
        if ply >= MAX_PLY - 1 {
            return if in_check { 0 } else { self.evaluate(pos, ply) };
        }

        let key = pos.hash();
        let mut tt_move = Move::NONE;
        let mut tt_eval = NO_EVAL;
        if let Some(entry) = self.tt.probe(key) {
            tt_move = entry.mv;
            tt_eval = entry.eval;
            let tt_score = score_from_tt(entry.score, ply);
            if !PV && bound_allows(entry.bound, tt_score, alpha, beta) {
                return tt_score;
            }
        }

        let static_eval;
        let mut best;
        if in_check {
            static_eval = NO_EVAL;
            best = -INFINITY;
        } else {
            static_eval = if tt_eval != NO_EVAL {
                tt_eval
            } else {
                self.evaluate(pos, ply)
            };
            best = static_eval;
            if best >= beta {
                return best;
            }
            if best > alpha {
                alpha = best;
            }
        }

        let mut list = MoveList::new();
        if in_check {
            generate::<GEN_ALL>(pos, &mut list);
            if list.is_empty() {
                return -MATE + ply as i32;
            }
        } else {
            generate::<GEN_TACTICAL>(pos, &mut list);
        }
        let mut scores = [0i32; MAX_MOVES];
        self.score_moves(pos, &list, tt_move, ply, &mut scores);

        let original_alpha = alpha;
        let mut best_move = Move::NONE;
        let len = list.len();
        for i in 0..len {
            let mut pick = i;
            for j in i + 1..len {
                if scores[j] > scores[pick] {
                    pick = j;
                }
            }
            list.swap(i, pick);
            scores.swap(i, pick);
            let mv = list[i];

            if !in_check && mv != tt_move && scores[i] < 0 {
                continue;
            }

            let child = self.make_move(pos, mv, ply);
            self.tt.prefetch(child.hash());
            self.push_hash(child.hash());
            let score = -self.qsearch::<PV>(&child, -beta, -alpha, ply + 1);
            self.pop_hash();
            if self.stopped {
                return 0;
            }

            if score > best {
                best = score;
                if score > alpha {
                    best_move = mv;
                    alpha = score;
                    if PV {
                        self.update_pv(ply, mv);
                    }
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }

        let bound = if best >= beta {
            Bound::Lower
        } else if best > original_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.tt.store(
            key,
            best_move,
            score_to_tt(best, ply),
            static_eval,
            0,
            bound,
        );
        best
    }

    /// Decide whether Lazy SMP helper thread `self.id` skips `depth` to diversify work.
    fn skips_depth(&self, depth: i32) -> bool {
        if self.id == 0 {
            return false;
        }
        let i = (self.id - 1) % SKIP_SIZE.len();
        ((depth + SKIP_PHASE[i]) / SKIP_SIZE[i]) % 2 != 0
    }

    /// Elapsed search time counting toward clock limits (zero while pondering, offset after `ponderhit`).
    fn limit_clock(&self) -> Duration {
        if self.shared.pondering.load(Ordering::Relaxed) {
            return Duration::ZERO;
        }
        let offset = Duration::from_millis(self.shared.ponder_offset_ms.load(Ordering::Relaxed));
        self.limits.start.elapsed().saturating_sub(offset)
    }

    /// Print a standard UCI `info` line for a completed iteration.
    fn report(&self, depth: i32, score: i32) {
        let millis = self.limits.start.elapsed().as_millis().max(1) as u64;
        let nodes = self.shared.nodes.load(Ordering::Relaxed) + (self.nodes - self.flushed);
        let mut line = format!(
            "info depth {depth} seldepth {} score {} nodes {nodes} nps {} hashfull {} time {millis} pv",
            self.seldepth,
            format_score(score),
            nodes * 1000 / millis,
            self.tt.hashfull(),
        );
        for mv in &self.pv[0][..self.pv_len[0]] {
            line.push(' ');
            line.push_str(&mv.to_string());
        }
        println!("{line}");
    }

    /// Iterative deepening + aspiration-window loop for one thread.
    fn iterate(&mut self, root: &Position, report: bool) -> SearchResult {
        let mut result = SearchResult::default();
        let root_moves = legal_moves(root);
        if root_moves.is_empty() {
            return result;
        }
        result.best_move = root_moves[0];

        let max_depth = self.limits.depth.clamp(1, MAX_PLY as i32 - 1);
        let mut previous = 0;

        for depth in 1..=max_depth {
            if self.skips_depth(depth) {
                continue;
            }
            self.root_depth = depth;

            let mut delta = self.params.aspiration_delta;
            let (mut alpha, mut beta) = if self.params.aspiration && depth >= 5 {
                (previous - delta, previous + delta)
            } else {
                (-INFINITY, INFINITY)
            };

            let score = loop {
                let score = self.search::<true>(root, alpha, beta, depth, 0);
                if self.stopped {
                    break score;
                }
                if score <= alpha {
                    beta = (alpha + beta) / 2;
                    alpha = (score - delta).max(-INFINITY);
                } else if score >= beta {
                    beta = (score + delta).min(INFINITY);
                } else {
                    break score;
                }
                delta += delta / 2;
                if delta > 500 {
                    alpha = -INFINITY;
                    beta = INFINITY;
                }
            };
            if self.stopped {
                break;
            }

            previous = score;
            result.score = score;
            result.depth = depth;
            if self.pv_len[0] > 0 {
                result.best_move = self.pv[0][0];
                result.ponder_move = if self.pv_len[0] > 1 {
                    self.pv[0][1]
                } else {
                    Move::NONE
                };
                result.pv = self.pv[0][..self.pv_len[0]].to_vec();
            }
            if report && self.id == 0 {
                self.report(depth, score);
            }

            if self.id == 0 && !self.limits.infinite {
                if let Some(soft) = self.limits.soft
                    && self.limit_clock() >= soft
                {
                    break;
                }
                if self.limits.soft.is_some()
                    && score.abs() >= MATE_IN_MAX
                    && MATE - score.abs() <= depth
                {
                    break;
                }
                if root_moves.len() == 1 && self.limits.soft.is_some() {
                    break;
                }
            }
        }

        self.flush_nodes();
        result.nodes = self.nodes;
        result
    }
}

/// Run a full search from `root`, blocking until completion.
///
/// Spawns one scoped Lazy SMP helper per extra entry in `heuristics`
/// (`heuristics[0]` is the main thread). In `infinite` or `ponder` mode, waits
/// after reaching max depth until `shared.stop` is raised.
#[allow(clippy::too_many_arguments)]
pub fn think(
    root: &Position,
    game_hist: &[u64],
    limits: &SearchLimits,
    tt: &TranspositionTable,
    shared: &SharedState,
    params: Params,
    heuristics: &mut [Box<Heuristics>],
    report: bool,
) -> SearchResult {
    let Some((main_heur, helpers)) = heuristics.split_first_mut() else {
        return SearchResult::default();
    };

    let mut result = std::thread::scope(|scope| {
        for (i, heur) in helpers.iter_mut().enumerate() {
            let id = i + 1;
            let builder = std::thread::Builder::new().stack_size(THREAD_STACK_BYTES);
            let _ = builder.spawn_scoped(scope, move || {
                let mut searcher =
                    Searcher::new(id, tt, shared, limits, params, heur, root, game_hist);
                searcher.iterate(root, false);
            });
        }

        let mut main = Searcher::new(0, tt, shared, limits, params, main_heur, root, game_hist);
        let result = main.iterate(root, report);

        while !shared.stop.load(Ordering::Relaxed)
            && (limits.infinite || shared.pondering.load(Ordering::Relaxed))
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        shared.stop.store(true, Ordering::Relaxed);
        result
    });

    result.nodes = shared.nodes.load(Ordering::Relaxed);
    result
}

/// Single-threaded fixed-depth search helper with a fresh `hash_mb` MiB TT and default [`Params`].
///
/// ```
/// use wustite::Position;
/// use wustite::search::search_depth;
///
/// let result = search_depth(&Position::startpos(), 5, 8);
/// assert!(!result.best_move.is_none());
/// assert_eq!(result.depth, 5);
/// ```
pub fn search_depth(pos: &Position, depth: i32, hash_mb: usize) -> SearchResult {
    let tt = TranspositionTable::new(hash_mb);
    let shared = SharedState::new();
    let limits = SearchLimits::depth_only(depth);
    let mut heuristics = vec![Heuristics::new()];
    think(
        pos,
        &[pos.hash()],
        &limits,
        &tt,
        &shared,
        Params::default(),
        &mut heuristics,
        false,
    )
}
