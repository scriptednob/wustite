//! The search: alpha-beta with principal variation search, iterative deepening and
//! Lazy SMP.
//!
//! Given a position and a set of limits, it looks
//! ahead through the game tree and returns the best move it can find. The public
//! entry points are [`think`] (full control, used by the UCI layer) and
//! [`search_depth`] (a convenience for fixed-depth searches). The tunable switches
//! live in [`Params`].
//!
//! # Overview
//!
//! ```text
//! think
//!  |-- spawns helper searchers (Lazy SMP), all sharing one TranspositionTable
//!  `-- main searcher: iterate
//!       for depth in 1..=max_depth            iterative deepening
//!         aspiration window around the last score
//!           search<PV = true>(root)           principal variation search
//!             search<PV = false>(...)         zero-window scouts
//!               qsearch(...)                  captures only, at depth 0
//! ```
//!
//! All scores are centipawns from the side to move's point of view (negamax),
//! bounded by [`INFINITY`]. Mate scores are `MATE - ply` (the sooner the mate,
//! the higher the score); anything at or beyond [`MATE_IN_MAX`] is a mate score
//! and is displayed by [`format_score`] as `mate N`.
//!
//! # Fail-soft negamax with PVS
//!
//! The core is a negamax alpha-beta search that is *fail-soft*: a node returns
//! the best score it actually found even when it falls outside the window, which
//! gives the transposition table tighter bounds than clamping to alpha/beta.
//!
//! It is a *principal variation search*: the first move at each node is searched
//! with the full window `(alpha, beta)`; every later move is first tried with a
//! cheap *zero-window* `(alpha, alpha + 1)` that only asks "is this move better
//! than the best so far?". Only if the answer is yes (and the node is on the
//! principal variation) is the move re-searched with the full window. Good move
//! ordering makes the first move the best most of the time, so the zero-window
//! scouts nearly always confirm that and cost little. The search is generic over
//! `const PV: bool`, so the compiler emits separate PV and non-PV versions with
//! the PV-only code removed from the latter.
//!
//! # Iterative deepening and aspiration windows
//!
//! [`think`] searches depth 1, then 2, then 3 and so on. Each iteration fills the transposition table and history tables that make the
//! next one faster, the best move is always available if time runs out, and the
//! time manager gets a clean point to decide whether to start another iteration.
//!
//! From depth 5 the root search uses an **aspiration window**: instead of
//! `(-INFINITY, INFINITY)` it searches `(previous - delta, previous + delta)` with
//! `delta` starting at [`Params::aspiration_delta`]. A narrow window cuts far more
//! of the tree. If the result falls outside the window the search is repeated with
//! the failing side widened (`delta` grows by half each time and the window opens
//! fully beyond 500).
//!
//! # Search enhancements
//!
//! Each of these has a switch in [`Params`] and a hidden UCI option (see
//! [`crate::uci`]). "Default" means on in [`Params::default`].
//!
//! **Pruning and reductions**
//!
//! | Technique | Default | What it does |
//! | --- | --- | --- |
//! | Check extension | always | A node in check searches one ply deeper. |
//! | Mate-distance pruning | always | Bounds alpha and beta by the shortest and longest possible mate from this ply. |
//! | Transposition-table cutoff | always | Non-PV nodes return a stored score when its depth and bound suffice. |
//! | Internal iterative reduction (`iir`) | on | With no table move at depth 4 or more, search one ply shallower. |
//! | Reverse futility pruning (`rfp`) | on | At depth 8 or less, if `eval - margin * depth >= beta`, return eval. |
//! | Razoring (`razoring`) | off | At depth 3 or less, if eval is far below alpha, verify with a quiescence search. |
//! | Null-move pruning (`nmp`) | on | Give the opponent a free move; if that still fails high, prune. Reduction `3 + depth/4 + min((eval-beta)/200, 3)`. Disabled without non-pawn material and directly after another null move. At depth 12 or more the result is verified by a reduced real search with null moves restricted. |
//! | ProbCut (`probcut`) | off | Try SEE-good captures against a raised beta; if one refutes it with a shallow search, prune. |
//! | Late move reductions (`lmr`) | on | Later quiet moves are searched shallower, by a logarithmic table, and re-searched if they surprise. |
//! | Late move pruning (`lmp`) | on | At depth 6 or less, skip quiet moves beyond `3 + depth^2` (half that when not improving). |
//! | Futility pruning (`futility`) | on | At depth 6 or less, skip quiet moves when `eval + 100 + 120 * depth <= alpha`. |
//! | SEE pruning (`seepruning`) | on | At depth 8 or less, skip moves that lose too much material by static exchange. |
//! | History pruning (`histpruning`) | off | Skip quiet moves with very negative history at low depth. |
//! | History-based LMR (`histlmr`) | off | Reduce less for moves with good history, more for bad. |
//!
//! **Extensions**
//!
//! | Technique | Default | What it does |
//! | --- | --- | --- |
//! | Singular extensions (`singular`) | on | If the table move is much better than all alternatives (verified by a reduced search that excludes it), extend it by one ply, or two in non-PV nodes when it is better by a wide margin. If even the alternatives beat the bound, return (multi-cut); if the table score already beats beta, search the move one ply shallower (negative extension). |
//!
//! **Evaluation and ordering**
//!
//! | Technique | Default | What it does |
//! | --- | --- | --- |
//! | TT-based evaluation (`tteval`) | on | A stored score that is a valid bound in the right direction replaces the static evaluation as the node's estimate. |
//! | Continuation history (`conthistory`) | on | History scores conditioned on the previous one and two moves. |
//! | Capture history (`capturehistory`) | off | History for captures by (piece, destination, victim). |
//!
//! Techniques marked off were implemented and tested but did not show a
//! measurable gain in match testing, and remain behind flags.
//!
//! Pruning and reductions are suspended when `alpha` is a mate score or `beta` is
//! a negated mate score, so that the shortest mate is always found.
//!
//! # Move ordering
//!
//! Alpha-beta is fast only if the best move is tried first. Moves are scored once
//! per node, and the highest-scoring unplayed move is picked each step (a
//! selection sort, since most nodes are cut off after one or two moves).
//!
//! | Priority | Class | Score |
//! | --- | --- | --- |
//! | 1 | Transposition-table move | `1 << 30` |
//! | 2 | Winning or equal captures (SEE >= 0) and queen promotions | `1 << 28` + MVV-LVA |
//! | 3 | First killer move | `(1 << 26) + 1` |
//! | 4 | Second killer move | `1 << 26` |
//! | 5 | Counter move | `1 << 25` |
//! | 6 | Other quiets | butterfly history plus continuation history |
//! | 7 | Losing captures and under-promotions | `-(1 << 24)` + MVV-LVA |
//!
//! *MVV-LVA* is "most valuable victim, least valuable attacker": captures are
//! ordered by victim value first, then cheapest attacker.
//! *Killer moves* are quiet moves that caused a beta cutoff at the same ply in a
//! sibling branch. The *counter move* is the reply that last refuted the
//! opponent's previous move. *History* tables count how often a quiet move
//! caused cutoffs; bonuses are applied with a *gravity* formula that
//! keeps entries bounded without explicit clamping.
//!
//! # Quiescence search
//!
//! At depth zero the search runs a quiescence search,
//! which keeps searching **captures and promotions only** until the position is
//! "quiet". The static evaluation acts as a *stand pat* score: the side to move is
//! assumed able to decline all captures, so the evaluation is a lower bound.
//! Captures that SEE says lose material are skipped. When in check, every
//! evasion is searched, because standing pat would be illegal.
//!
//! # Draws
//!
//! A position scores 0 when it is a draw by insufficient material, by the
//! fifty-move rule (the clock reaches 100, unless the side to move is checkmated),
//! or by repetition. Repetition is detected against both the game history sent
//! by the GUI and the current search path: a position repeated within the search
//! counts as a draw on its first recurrence, while one that only occurred in the
//! game history before the root must already have occurred twice.
//!
//! # Lazy SMP
//!
//! With `Threads > 1`, [`think`] spawns one extra searcher thread per helper.
//! Every thread searches the same root with its own stack, history tables and
//! accumulators, and they cooperate **only** through the shared transposition
//! table. This is "Lazy SMP". Helper threads skip some depths in a staggered
//! pattern so they do not all search identical iterations, and the table steers
//! them into different parts of the tree. Only the main thread (id 0) decides
//! when to stop and reports results.
//!
//! # Time and stopping
//!
//! Every 1024 nodes a searcher flushes its node count into [`SharedState`] and
//! checks the stop flag. The main thread also checks the node limit and the hard
//! time limit. Pondering is supported: while [`SharedState::pondering`] is set the
//! clock does not run, and on `ponderhit` the time already spent is credited
//! through [`SharedState::ponder_offset_ms`].
//!
//! # Examples
//!
//! A fixed-depth search, finding a mate in one:
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
//! Searching with explicit limits and options:
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

/// The deepest ply the search will reach, including quiescence. Sizes the per-thread stack arrays.
pub const MAX_PLY: usize = 128;
/// Larger than any real score. The initial alpha-beta window is `(-INFINITY, INFINITY)`.
pub const INFINITY: i32 = 32_000;
/// The score of being checkmated at ply 0, negated for the opponent. A mate in N plies scores `MATE - N`, so shorter mates score higher.
pub const MATE: i32 = 31_000;
/// Scores at or above this value are mate scores (the longest mate the search can see is `MAX_PLY` plies). Scores at or below its negation are mated-in scores.
pub const MATE_IN_MAX: i32 = MATE - MAX_PLY as i32;

/// Sentinel for "no static evaluation", stored when the side to move is in check. Just outside the score range.
const NO_EVAL: i32 = 32_001;
/// Capacity of the repetition-detection history: game positions plus the search path.
const HIST_CAP: usize = 512;
/// The saturation point of the history tables. Used in the gravity formula.
const HISTORY_MAX: i32 = 16_384;
/// Stack size for helper threads. Each searcher holds large fixed-size arrays on its stack, so the default thread stack is not enough.
const THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Ordering score of the transposition-table move: always first.
const TT_MOVE_SCORE: i32 = 1 << 30;
/// Base ordering score of captures with non-negative SEE, and queen promotions.
const GOOD_CAPTURE: i32 = 1 << 28;
/// Base ordering score of killer moves.
const KILLER_SCORE: i32 = 1 << 26;
/// Ordering score of the counter move.
const COUNTER_SCORE: i32 = 1 << 25;
/// Base ordering score of losing captures and under-promotions. Negative, so they sort after all quiet moves.
const BAD_CAPTURE: i32 = -(1 << 24);

/// Lazy SMP depth-skipping pattern, part 1: the block length for helper thread `i`.
const SKIP_SIZE: [i32; 20] = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4];
/// Lazy SMP depth-skipping pattern, part 2: the phase offset for helper thread `i`. Together with [`SKIP_SIZE`] it makes different helpers skip different depths.
const SKIP_PHASE: [i32; 20] = [0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7];

/// Natural logarithm computed in `const` context, by range reduction to `[1, 2)`
/// and an atanh series. Needed because floating-point `ln` is not `const`, and
/// the reduction table is built at compile time.
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

/// Builds the late-move-reduction table: `reduction = 0.75 + ln(depth) * ln(move_number) / 2.25`,
/// truncated to an integer, for depth and move number below 64.
///
/// The formula is logarithmic in both arguments: late moves at high depth are
/// reduced the most, early moves and shallow nodes barely at all.
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

/// The reduction table, indexed `[depth][move_number]`.
static LMR_TABLE: [[u8; 64]; 64] = build_lmr();

/// Switches and tuning values for the search.
///
/// Every enhancement can be turned off (see [`Params::baseline`]) so its
/// strength contribution can be measured by self-play. [`Params::default`] is the
/// configuration the engine ships with. Through UCI, the fields are set with
/// `setoption name <name> value <number>`; see [`Params::set`] for the names.
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
    /// Aspiration windows at the root from depth 5.
    pub aspiration: bool,
    /// Late move reductions.
    pub lmr: bool,
    /// Null-move pruning.
    pub nmp: bool,
    /// Reverse futility pruning (also called static null-move pruning).
    pub rfp: bool,
    /// Razoring: verify hopeless nodes with a quiescence search at shallow depth.
    pub razoring: bool,
    /// Futility pruning of quiet moves.
    pub futility: bool,
    /// Late move pruning of quiet moves.
    pub lmp: bool,
    /// Pruning of moves with a bad static exchange result.
    pub see_pruning: bool,
    /// Internal iterative reduction when there is no table move.
    pub iir: bool,
    /// Capture history in move ordering and updates.
    pub capture_history: bool,
    /// Continuation history (one and two plies back) for quiet move ordering.
    pub cont_history: bool,
    /// Singular extensions, double extensions, multi-cut and negative extensions.
    pub singular: bool,
    /// ProbCut.
    pub probcut: bool,
    /// Use a valid transposition-table score as the node's evaluation estimate.
    pub tt_eval: bool,
    /// Adjust late-move reductions by the move's history score.
    pub hist_lmr: bool,
    /// Evaluate with the NNUE network (`true`) or the classical PeSTO evaluation (`false`).
    pub nnue: bool,
    /// Prune quiet moves with very negative history at low depth.
    pub hist_pruning: bool,
    /// History pruning threshold: prune when history is below `-margin * depth`.
    pub hist_prune_margin: i32,
    /// Reverse futility margin per ply of remaining depth, in centipawns.
    pub rfp_margin: i32,
    /// Razoring margin per ply of remaining depth, in centipawns.
    pub razor_margin: i32,
    /// Constant part of the futility margin, in centipawns.
    pub futility_base: i32,
    /// Per-ply part of the futility margin, in centipawns.
    pub futility_slope: i32,
    /// Base null-move reduction in plies. The depth- and eval-dependent parts are added to it.
    pub nmp_base: i32,
    /// Null-move results at or beyond this depth are verified by a reduced real search.
    pub nmp_verify_depth: i32,
    /// SEE pruning margin for captures: skip when SEE is below `-margin * depth`.
    pub see_capture_margin: i32,
    /// SEE pruning margin for quiet moves: skip when SEE is below `-margin * depth`.
    pub see_quiet_margin: i32,
    /// Initial half-width of the aspiration window, in centipawns.
    pub aspiration_delta: i32,
    /// How far above beta ProbCut tests, in centipawns.
    pub probcut_margin: i32,
    /// Minimum depth at which singular extensions are tried.
    pub singular_depth: i32,
}

impl Params {
    /// The plain configuration: every pruning and reduction feature off, NNUE on.
    ///
    /// Useful as the reference when measuring what one feature gains. The numeric
    /// margins hold their normal values, so enabling a single flag gives that feature
    /// with its standard tuning.
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

    /// Sets one parameter by name. Returns `false` if the name is not recognised.
    ///
    /// Names are case-insensitive and contain no spaces. Boolean parameters take
    /// `0` for off and any other value for on; numeric parameters store the value.
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

/// What bounds a search: a depth, a node budget, and soft and hard time limits.
///
/// Build one from a UCI `go` command with [`crate::time::build_limits`], or use
/// [`SearchLimits::depth_only`] for a plain depth-limited search.
#[derive(Clone, Copy, Debug)]
pub struct SearchLimits {
    /// Maximum iteration depth in plies.
    pub depth: i32,
    /// Maximum nodes to search, or `0` for no limit.
    pub nodes: u64,
    /// Soft time limit: checked between iterations, after which no new iteration starts.
    pub soft: Option<Duration>,
    /// Hard time limit: checked inside the search, after which the search aborts.
    pub hard: Option<Duration>,
    /// Search until told to stop, ignoring the time limits.
    pub infinite: bool,
    /// When the search started. All elapsed time is measured from here.
    pub start: Instant,
}

impl SearchLimits {
    /// Limits that only bound the depth, with no node or time limit.
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

/// State shared between the UCI thread and every search thread, all atomics.
///
/// The UCI thread sets [`SharedState::stop`] to end a search and
/// [`SharedState::pondering`] to run one on the opponent's time; searchers add to
/// the node counter and read both flags.
pub struct SharedState {
    /// Set to ask all searchers to finish as soon as possible.
    pub stop: AtomicBool,
    /// Total nodes searched by all threads, updated in batches of about 1024.
    pub nodes: AtomicU64,
    /// While `true`, the search clock does not run and the search does not stop on its own.
    pub pondering: AtomicBool,
    /// Milliseconds already elapsed when `ponderhit` arrived. Subtracted from the elapsed time so only time after the hit counts.
    pub ponder_offset_ms: AtomicU64,
}

impl SharedState {
    /// A fresh state: not stopped, no nodes, not pondering.
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

/// Entries in the continuation-history table: `12 * 64 * 12 * 64` (previous piece and square by current piece and square).
const CONT_SIZE: usize = 12 * 64 * 12 * 64;

/// Per-thread move-ordering tables that persist between searches within a game.
///
/// * **Butterfly history** `[colour][from][to]`: how often a quiet move caused
///   a cutoff.
/// * **Counter moves** `[piece][to]`: the reply that refuted the opponent's last move.
/// * **Capture history** `[piece][to][victim]`: the capture analogue.
/// * **Continuation history**: quiet-move score conditioned on the move one or two
///   plies earlier.
///
/// They are created with [`Heuristics::new`], which is boxed because the
/// continuation table is large, and wiped with [`Heuristics::clear`] on a new
/// game. Each search thread needs its own set.
pub struct Heuristics {
    butterfly: [[[i16; 64]; 64]; 2],
    counter: [[Move; 64]; 12],
    capture: [[[i16; 6]; 64]; 12],
    cont: Box<[i16]>,
}

impl Heuristics {
    /// Allocates zeroed tables.
    pub fn new() -> Box<Heuristics> {
        Box::new(Heuristics {
            butterfly: [[[0; 64]; 64]; 2],
            counter: [[Move::NONE; 64]; 12],
            capture: [[[0; 6]; 64]; 12],
            cont: vec![0i16; CONT_SIZE].into_boxed_slice(),
        })
    }

    /// Zeroes all tables.
    pub fn clear(&mut self) {
        self.butterfly = [[[0; 64]; 64]; 2];
        self.counter = [[Move::NONE; 64]; 12];
        self.capture = [[[0; 6]; 64]; 12];
        self.cont.fill(0);
    }
}

/// Flattens a (previous piece, previous destination, piece, destination) tuple into an index into the continuation-history array.
#[inline(always)]
fn cont_index(prev_piece: usize, prev_to: usize, piece: usize, to: usize) -> usize {
    ((prev_piece * 64 + prev_to) * 12 + piece) * 64 + to
}

/// Applies a history bonus with *gravity*: `entry += bonus - entry * |bonus| / HISTORY_MAX`.
///
/// The correction term shrinks the update as the entry approaches
/// [`HISTORY_MAX`], so values stay bounded without a hard clamp and old results
/// decay as new information comes in.
#[inline(always)]
fn gravity(entry: &mut i16, bonus: i32) {
    let current = *entry as i32;
    *entry = (current + bonus - current * bonus.abs() / HISTORY_MAX) as i16;
}

/// The `(piece, destination, victim)` index of a capture in the capture-history table, or `None` for non-captures.
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

/// The outcome of a [`think`] call.
///
/// When the search was stopped before depth 1 completed, `best_move` is the first
/// legal move so that a legal move is always available. With no legal moves
/// (checkmate or stalemate) it is [`Move::NONE`](crate::Move::NONE).
#[derive(Clone, Debug, Default)]
pub struct SearchResult {
    /// The move to play.
    pub best_move: Move,
    /// The expected reply, second in the principal variation, for UCI pondering. [`Move::NONE`](crate::Move::NONE) when unknown.
    pub ponder_move: Move,
    /// Score of the last completed iteration. Format it with [`format_score`].
    pub score: i32,
    /// Depth of the last completed iteration.
    pub depth: i32,
    /// Total nodes searched by all threads.
    pub nodes: u64,
    /// The principal variation: the line both sides are expected to play.
    pub pv: Vec<Move>,
}

/// Per-ply search state: the static evaluation (for the *improving* test), the
/// two killer moves, the move that was made to reach the next ply (for
/// continuation history and counter moves), the piece that moved, and the move
/// excluded during a singular-extension verification search.
#[derive(Clone, Copy)]
struct StackEntry {
    static_eval: i32,
    killers: [Move; 2],
    mv: Move,
    piece: usize,
    excluded: Move,
}

/// A cleared [`StackEntry`].
const EMPTY_ENTRY: StackEntry = StackEntry {
    static_eval: NO_EVAL,
    killers: [Move::NONE; 2],
    mv: Move::NONE,
    piece: 0,
    excluded: Move::NONE,
};

/// Adjusts a mate score before it goes into the transposition table.
///
/// Mate scores are relative to the root (`MATE - ply`). Stored as is, the same
/// position reached at a different ply would report a different mate distance.
/// The table instead stores the distance from the *node*, by adding the current
/// ply to winning mates and subtracting it from losing ones.
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

/// The inverse of [`score_to_tt`]: re-bases a stored mate score onto the current ply.
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

/// Whether a stored bound proves the node's result is outside or at the edge of the window: exact always, a lower bound when it reaches beta, an upper bound when it does not exceed alpha.
#[inline(always)]
fn bound_allows(bound: Bound, score: i32, alpha: i32, beta: i32) -> bool {
    match bound {
        Bound::Exact => true,
        Bound::Lower => score >= beta,
        Bound::Upper => score <= alpha,
        Bound::None => false,
    }
}

/// Formats a score as UCI does: `cp <centipawns>` or `mate <moves>`, where the
/// mate distance is in *moves* (not plies) and is negative when the side to move
/// is being mated.
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

/// Ordering score for a capture or promotion: MVV-LVA (`victim * 16 - attacker`)
/// plus a small capture-history term, offset into the good-capture or
/// bad-capture range by SEE. Queen promotions rank above everything else in the
/// good range; under-promotions are treated as bad.
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

/// All state of one search thread.
///
/// A `Searcher` is created per thread for every call to [`think`], borrowing the
/// shared table, the shared flags and its own [`Heuristics`]. Its large arrays
/// (the ply stack, principal variation triangle, repetition history and NNUE
/// accumulators) are why helper threads need a large stack.
///
/// The `pv` array is a *triangular* table: each ply's principal variation is
/// built by prepending the move just played to its child's, so the root's
/// variation is complete when the search returns.
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
    /// Creates a searcher for thread `id` (0 is the main thread).
    ///
    /// Copies the relevant tail of the game history for repetition detection,
    /// appends the root hash if it is missing, and, when NNUE is enabled, loads the
    /// active network and computes the root accumulator.
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

    /// Static evaluation of the node at `ply`.
    ///
    /// Bare-king endings use the mop-up rule from [`crate::eval`]. Otherwise the
    /// network is applied to the accumulator for this ply, with the result clamped
    /// inside the non-mate score range, or the classical evaluation is used if NNUE
    /// is off. Debug builds assert that the accumulator equals a fresh recomputation.
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

    /// Writes the accumulator of the child reached by `mv` into the next ply's slot.
    ///
    /// This must happen immediately before descending into the child. In particular,
    /// singular-extension verification searches run at the **same** ply and write
    /// their own child accumulators, so the real child's update must come after the
    /// verification search, otherwise the child would search with a corrupt
    /// accumulator.
    #[inline(always)]
    fn update_accumulator(&mut self, pos: &Position, mv: Move, ply: usize) {
        if let Some(net) = &self.net {
            let (head, tail) = self.acc.split_at_mut(ply + 1);
            nnue::update(net, &head[ply], &mut tail[0], pos, mv);
        }
    }

    /// Makes `mv` on a copy of `pos` and updates the NNUE accumulator for the child.
    #[inline(always)]
    fn make_move(&mut self, pos: &Position, mv: Move, ply: usize) -> Position {
        self.update_accumulator(pos, mv, ply);
        pos.make_move(mv)
    }

    /// Makes a null move and copies the parent's accumulator, since no features change.
    #[inline(always)]
    fn make_null_move(&mut self, pos: &Position, ply: usize) -> Position {
        if self.net.is_some() {
            self.acc[ply + 1] = self.acc[ply];
        }
        pos.make_null_move()
    }

    /// Adds the nodes counted since the last flush to the shared counter.
    fn flush_nodes(&mut self) {
        let delta = self.nodes - self.flushed;
        self.flushed = self.nodes;
        self.shared.nodes.fetch_add(delta, Ordering::Relaxed);
    }

    /// Sets the shared stop flag and stops this searcher.
    fn trigger_stop(&mut self) {
        self.stopped = true;
        self.shared.stop.store(true, Ordering::Relaxed);
    }

    /// Called every 1024 nodes. Flushes the node count, and then checks whether to
    /// stop: the shared stop flag for every thread, and the node and hard-time limits
    /// for the main thread. Nothing stops before depth 2 is reached, so a move always
    /// exists.
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

    /// Whether `pos` is a draw: insufficient material, the fifty-move rule, or a
    /// repetition.
    ///
    /// Only positions an even number of plies back, at least four, can repeat the
    /// current one, and only back to the last irreversible move (the halfmove
    /// clock). A repetition inside the search tree (within `ply` plies of the root)
    /// counts at once, because the side can choose to repeat; one that reaches before
    /// the root must have occurred twice already. At 100 halfmoves the position is
    /// a draw unless the side to move is checkmated.
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

    /// Pushes a position hash onto the repetition history when descending.
    #[inline(always)]
    fn push_hash(&mut self, key: u64) {
        self.hist[self.hist_len] = key;
        self.hist_len += 1;
    }

    /// Pops the hash pushed by [`Searcher::push_hash`] when returning.
    #[inline(always)]
    fn pop_hash(&mut self) {
        self.hist_len -= 1;
    }

    /// Builds the principal variation at `ply` as `mv` followed by the child's variation.
    fn update_pv(&mut self, ply: usize, mv: Move) {
        self.pv[ply][0] = mv;
        let child_len = self.pv_len[ply + 1];
        for i in 0..child_len {
            self.pv[ply][i + 1] = self.pv[ply + 1][i];
        }
        self.pv_len[ply] = child_len + 1;
    }

    /// Updates the butterfly history entry for a quiet move by `bonus` (negative for a penalty).
    fn update_history(&mut self, color: Color, mv: Move, bonus: i32) {
        let entry = &mut self.heur.butterfly[color.idx()][mv.from_sq().idx()][mv.to_sq().idx()];
        gravity(entry, bonus);
    }

    /// Updates the continuation-history entries pairing the current move with the moves one and two plies earlier.
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

    /// The total history score of a quiet move: butterfly history plus, when enabled, continuation history one and two plies back.
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

    /// The capture-history score of a capture, or `0` when that feature is off.
    fn capture_history(&self, pos: &Position, mv: Move) -> i32 {
        if !self.params.capture_history {
            return 0;
        }
        match capture_slot(pos, mv) {
            Some((piece, to, victim)) => self.heur.capture[piece][to][victim] as i32,
            None => 0,
        }
    }

    /// Updates one capture-history entry by `bonus`.
    fn reward_capture(&mut self, pos: &Position, mv: Move, bonus: i32) {
        if let Some((piece, to, victim)) = capture_slot(pos, mv) {
            gravity(&mut self.heur.capture[piece][to][victim], bonus);
        }
    }

    /// Assigns an ordering score to every move in `list` according to the table in the [module documentation](crate::search).
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

    /// The main recursive alpha-beta search.
    ///
    /// `PV` marks nodes on the principal variation. `depth` is the remaining depth
    /// in plies and `ply` the distance from the root. Returns the node's score, or 0
    /// when the search has been stopped (the caller discards it).
    ///
    /// The order of operations at a node:
    ///
    /// 1. **Check extension**, then drop into [`Searcher::qsearch`] if depth is zero.
    /// 2. **Bookkeeping**: node count, periodic [`Searcher::poll`], selective depth.
    /// 3. **Draw detection**, the ply limit, and **mate-distance pruning** (not at
    ///    the root).
    /// 4. **Transposition-table probe**. A deep enough entry with a usable bound ends
    ///    a non-PV node immediately.
    /// 5. **Static evaluation** (not when in check), taken from the table when
    ///    cached, and the *improving* flag: whether the evaluation is better than two
    ///    plies ago, which makes pruning more or less aggressive.
    /// 6. **TT-based evaluation refinement**, then **internal iterative reduction**.
    /// 7. In non-PV nodes that are not in check: **razoring**, **reverse futility
    ///    pruning**, **null-move pruning** (with verification) and **ProbCut**.
    /// 8. **Generate and order** all legal moves. No moves means checkmate
    ///    (`-MATE + ply`) or stalemate (0).
    /// 9. For each move, best first:
    ///    * **SEE pruning**, **late move pruning**, **futility pruning** and
    ///      **history pruning**, only for non-root quiet or losing moves when no mate
    ///      score is in the window;
    ///    * **singular extension** of the table move;
    ///    * make the move, then **PVS**: full window for the first move, then a
    ///      zero-window search with a **late-move reduction** for later moves, a
    ///      re-search at full depth if the reduced search beat alpha, and a full-window
    ///      re-search if a PV node's score landed inside the window;
    ///    * on a new best score update alpha and the principal variation, and on a
    ///      beta cutoff reward the move in the history tables and killer slots and
    ///      penalise the quiet moves tried before it.
    /// 10. **Store** the result in the transposition table with a [`Bound`](crate::tt::Bound)
    ///     (lower bound if it failed high, exact if it improved alpha, upper bound
    ///     otherwise). Nothing is stored while a singular verification search is
    ///     running, since its result excludes a move.
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

    /// On a beta cutoff, raises the capture-history entry of the winning capture and lowers those of the captures tried before it.
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

    /// On a beta cutoff by a quiet move: records it as the newest killer, raises its
    /// butterfly and continuation history by a depth-scaled bonus
    /// (`155 * depth - 80`, clamped to `0..=2000`), lowers the history of the quiet
    /// moves tried earlier by the same amount, and records it as the counter move to
    /// the opponent's previous move.
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

    /// Quiescence search: resolves captures at the leaves so the static evaluation
    /// is only trusted in quiet positions.
    ///
    /// Without it the search would stop in the middle of an exchange and mistake a
    /// queen-for-pawn trade for a free pawn. Not in check, the static evaluation is a
    /// *stand-pat* lower bound (the side to move may decline every capture), and only
    /// captures and promotions are generated, skipping those with a negative ordering
    /// score (losing SEE) unless it is the table move. In check, every evasion is
    /// generated and there is no stand-pat; no legal moves is checkmate.
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

    /// Lazy SMP helper staggering. The main thread never skips. Helper `i` skips
    /// depth `d` when `((d + SKIP_PHASE[i]) / SKIP_SIZE[i]) % 2 == 1`, so that
    /// different helpers work on different depths and fill the shared table with
    /// complementary information.
    fn skips_depth(&self, depth: i32) -> bool {
        if self.id == 0 {
            return false;
        }
        let i = (self.id - 1) % SKIP_SIZE.len();
        ((depth + SKIP_PHASE[i]) / SKIP_SIZE[i]) % 2 != 0
    }

    /// Elapsed search time for limit checks: zero while pondering, and otherwise measured from the start minus any time already spent when `ponderhit` arrived.
    fn limit_clock(&self) -> Duration {
        if self.shared.pondering.load(Ordering::Relaxed) {
            return Duration::ZERO;
        }
        let offset = Duration::from_millis(self.shared.ponder_offset_ms.load(Ordering::Relaxed));
        self.limits.start.elapsed().saturating_sub(offset)
    }

    /// Prints a UCI `info` line: depth, selective depth, score, nodes, nodes per
    /// second, hash fullness, time and the principal variation.
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

    /// Iterative deepening driver for one thread. See the
    /// [module documentation](crate::search) for the aspiration-window loop.
    ///
    /// After each completed iteration it records the best move, ponder move, score
    /// and principal variation. On the main thread it then decides whether to stop:
    /// the soft time limit has passed, a forced mate within the searched depth has
    /// been found, or there is only one legal move.
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

/// Runs a complete search from `root` and returns the result.
///
/// * `game_hist` holds the hashes of every position of the game so far,
///   oldest first, for repetition detection. Pass `&[root.hash()]` for a
///   position with no history.
/// * `limits` bounds the search, `shared` carries the stop flag, ponder state and
///   node counter, and `tt` is the shared transposition table.
/// * `heuristics` supplies one [`Heuristics`] per thread, and its length **is the
///   thread count**: element 0 is the main thread and the rest are Lazy SMP
///   helpers. An empty slice returns a default result.
/// * `report` selects whether the main thread prints UCI `info` lines.
///
/// The call blocks until the search ends and uses scoped threads, so helpers are
/// always joined before it returns. When `limits.infinite` is set or pondering
/// is active, the call keeps waiting after the main thread finishes until
/// `shared.stop` is set, as the UCI protocol requires.
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

/// Searches `pos` to a fixed `depth` with a fresh `hash_mb` MiB table, one thread,
/// the default [`Params`] and no output. A convenience for tests, examples and
/// tools; the UCI layer uses [`think`] directly.
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
