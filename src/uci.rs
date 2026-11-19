//! The Universal Chess Interface front end.
//!
//! UCI is the text protocol chess GUIs and tournament managers use to talk to
//! engines: the GUI writes commands to the engine's standard input and the engine
//! answers on standard output, one command per line. [`run`] implements the engine
//! side and is what the `wustite` binary does when started without arguments.
//!
//! # Threads
//!
//! The thread that calls [`run`] only reads standard input and dispatches commands;
//! it never searches. `go` spawns a **worker thread** (with a 64 MiB stack, because
//! a searcher keeps large arrays on its stack) that calls
//! [`think`], which in turn spawns the Lazy SMP helpers. The
//! two communicate through the atomics in [`SharedState`]:
//! `stop` and `ponderhit` only flip flags, so they are handled immediately even
//! in the middle of a long search. The worker prints `bestmove` itself when the
//! search ends, and hands its history tables back to the UCI thread when it is
//! joined, so move-ordering knowledge survives from move to move.
//!
//! # Commands the engine understands
//!
//! | Command | Behaviour |
//! | --- | --- |
//! | `uci` | Prints `id` lines, every option, then `uciok`. |
//! | `isready` | Prints `readyok`. |
//! | `ucinewgame` | Stops any search, clears the hash table and history tables, resets the position. |
//! | `position startpos [moves ...]` | Sets the starting position and plays the given moves. |
//! | `position fen <fen> [moves ...]` | Sets a position from FEN and plays the given moves. |
//! | `go ...` | Starts a search. See below. |
//! | `stop` | Ends the search; `bestmove` is printed. |
//! | `ponderhit` | The opponent played the predicted move: the pondering search becomes a normal timed search. |
//! | `setoption name <name> [value <value>]` | Changes an option. See below. |
//! | `quit` | Stops the search and exits. |
//!
//! A move that is not legal ends the `moves` list with `info string illegal move`
//! and the position keeps the moves played so far. Unknown commands are ignored,
//! as the specification requires. End of input behaves like `quit`.
//!
//! Every position sent keeps its **game history** (the hash of each position
//! reached), which the search uses to see threefold repetition.
//!
//! ## `go` parameters
//!
//! `wtime`, `btime`, `winc`, `binc`, `movestogo`, `movetime`, `depth`, `nodes`,
//! `infinite` and `ponder`; see [`crate::time`] for how they become limits.
//! `go perft <n>` is a non-standard extension that prints the per-move node counts
//! and the total, like Stockfish.
//!
//! ## Output
//!
//! During a search the main thread prints one `info` line per completed
//! iteration:
//!
//! ```text
//! info depth 12 seldepth 18 score cp 31 nodes 184233 nps 1583000 hashfull 12 time 116 pv e2e4 e7e5 g1f3 ...
//! ```
//!
//! and when it finishes `bestmove <move>` followed by ` ponder <move>` when a
//! predicted reply is available. A position with no legal move prints
//! `bestmove 0000`. The score is `cp <centipawns>` or `mate <moves>`.
//!
//! # Options
//!
//! | Name | Type | Default | Range | Meaning |
//! | --- | --- | --- | --- | --- |
//! | `Hash` | spin | 16 | 1 to 65536 | Transposition table size in MiB. Changing it allocates a new table. |
//! | `Threads` | spin | 1 | 1 to 256 | Number of search threads (Lazy SMP). |
//! | `Move Overhead` | spin | 30 | 0 to 5000 | Milliseconds held back per move for communication delays. |
//! | `Ponder` | check | false | | Tells the GUI the engine supports pondering. |
//! | `UseNNUE` | check | true | | Evaluate with the network; when off, the classical evaluation is used. |
//! | `EvalFile` | string | `<internal>` | | Path of a network file, or `<internal>` for the built-in one. |
//! | `Clear Hash` | button | | | Empties the transposition table. |
//!
//! Option names are case-insensitive. Changing `Hash`, `UseNNUE` or `EvalFile`
//! first stops the search, and switching the evaluator also clears the table,
//! since stored scores came from a different evaluation.
//!
//! ## Hidden tuning options
//!
//! Every search switch and margin in [`Params`] can also be
//! set with `setoption` even though it is not advertised in the `uci` reply:
//! `lmr`, `nmp`, `rfp`, `futility`, `lmp`, `seepruning`, `aspiration`, `iir`,
//! `razoring`, `capturehistory`, `conthistory`, `singular`, `probcut`, `tteval`,
//! `histlmr`, `histpruning`, and the numeric margins `rfpmargin`, `razormargin`,
//! `futilitybase`, `futilityslope`, `nmpbase`, `nmpverifydepth`,
//! `seecapturemargin`, `seequietmargin`, `aspirationdelta`, `probcutmargin`,
//! `singulardepth` and `histprunemargin`.
//!
//! The value must be a **number**: `0` for off, anything else for on. `true` and
//! `false` are ignored for these options. This is how matches A/B-test a feature,
//! for example `setoption name NMP value 0`.
//!
//! # Debugging commands (not part of UCI)
//!
//! | Command | Behaviour |
//! | --- | --- |
//! | `d` | Prints the board, the FEN and the Zobrist key. |
//! | `eval` | Prints the static evaluation of the current position. |
//! | `bench [depth]` | Runs the benchmark, see [`crate::bench`]. |
//! | `go perft <n>` | Prints the move-by-move perft counts. |
//!
//! # Pondering
//!
//! With `Ponder` enabled the GUI sends `go ponder` right after the engine's move,
//! with the position after the engine's move **and** its predicted reply. The
//! engine searches without a clock. If the opponent plays the predicted move the
//! GUI sends `ponderhit`; the elapsed time is recorded and the search continues
//! with a normal clock that starts at the hit. If not, it sends `stop` and then a
//! fresh `position` and `go`.
//!
//! # A session
//!
//! ```text
//! > uci
//! < id name Wustite 0.3.0
//! < id author Taheem
//! < option name Hash type spin default 16 min 1 max 65536
//! < ...
//! < uciok
//! > isready
//! < readyok
//! > position startpos moves e2e4 e7e5
//! > go movetime 1000
//! < info depth 1 seldepth 1 score cp 25 nodes 30 nps ... pv g1f3
//! < ...
//! < bestmove g1f3 ponder b8c6
//! > quit
//! ```

use std::io::{self, BufRead};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;
use std::time::Instant;

use crate::bench;
use crate::board::Position;
use crate::eval::{bare_king_eval, evaluate};
use crate::movegen::parse_move;
use crate::nnue;
use crate::perft::divide;
use crate::search::{Heuristics, Params, SharedState, think};
use crate::time::{GoParams, build_limits};
use crate::tt::TranspositionTable;

/// Stack size of the search worker thread. The searcher stores large fixed-size arrays on its stack.
const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;
/// Default transposition table size in MiB.
const DEFAULT_HASH_MB: usize = 16;
/// Largest accepted `Hash` value in MiB.
const MAX_HASH_MB: usize = 65_536;
/// Largest accepted `Threads` value.
const MAX_THREADS: usize = 256;

/// Everything the UCI loop keeps between commands: the current position and game
/// history, the shared table, the shared search state, one [`Heuristics`](crate::search::Heuristics)
/// set per thread, the option values, and the handle of the running search worker.
///
/// The history tables are *moved into* the worker when a search starts and moved
/// back out when it is joined, which is why `heuristics` is empty while a search
/// runs.
struct Engine {
    pos: Position,
    game_hist: Vec<u64>,
    tt: Arc<TranspositionTable>,
    shared: Arc<SharedState>,
    heuristics: Vec<Box<Heuristics>>,
    threads: usize,
    overhead_ms: u64,
    params: Params,
    worker: Option<JoinHandle<Vec<Box<Heuristics>>>>,
    infinite: bool,
    search_start: Instant,
}

impl Engine {
    /// An engine in the starting position with default options.
    fn new() -> Engine {
        let pos = Position::startpos();
        Engine {
            game_hist: vec![pos.hash()],
            pos,
            tt: Arc::new(TranspositionTable::new(DEFAULT_HASH_MB)),
            shared: Arc::new(SharedState::new()),
            heuristics: vec![Heuristics::new()],
            threads: 1,
            overhead_ms: 30,
            params: Params::default(),
            worker: None,
            infinite: false,
            search_start: Instant::now(),
        }
    }

    /// Waits for the search worker to finish and takes back its history tables. If the worker panicked, fresh tables are created.
    fn join_worker(&mut self) {
        if let Some(handle) = self.worker.take() {
            match handle.join() {
                Ok(heuristics) => self.heuristics = heuristics,
                Err(_) => self.heuristics = vec![Heuristics::new()],
            }
        }
    }

    /// Raises the stop flag, cancels pondering and joins the worker.
    fn stop_search(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.pondering.store(false, Ordering::Relaxed);
        self.join_worker();
    }

    /// Handles `ponderhit`: records how long the ponder search already ran and switches it to a normal clock.
    fn ponder_hit(&mut self) {
        if self.shared.pondering.load(Ordering::Relaxed) {
            let offset = self.search_start.elapsed().as_millis() as u64;
            self.shared
                .ponder_offset_ms
                .store(offset, Ordering::Relaxed);
            self.shared.pondering.store(false, Ordering::Relaxed);
        }
    }

    /// Executes one input line. Returns `false` when the engine should exit (`quit`).
    fn handle(&mut self, line: &str) -> bool {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some((&command, rest)) = tokens.split_first() else {
            return true;
        };
        match command {
            "uci" => self.uci(),
            "isready" => println!("readyok"),
            "ucinewgame" => {
                self.stop_search();
                self.tt.clear();
                for heur in &mut self.heuristics {
                    heur.clear();
                }
                self.pos = Position::startpos();
                self.game_hist = vec![self.pos.hash()];
            }
            "position" => {
                self.join_worker();
                self.position(rest);
            }
            "go" => self.go(rest),
            "stop" => {
                self.shared.stop.store(true, Ordering::Relaxed);
                self.shared.pondering.store(false, Ordering::Relaxed);
            }
            "ponderhit" => self.ponder_hit(),
            "setoption" => self.setoption(rest),
            "d" => println!("{}", self.pos),
            "eval" => println!("static eval {}", self.static_eval()),
            "bench" => {
                let depth = rest.first().and_then(|d| d.parse().ok()).unwrap_or(10);
                let report = bench::run(depth, self.params, true);
                println!(
                    "{} nodes {} nps",
                    report.nodes,
                    (report.nodes as f64 / report.seconds.max(1e-9)) as u64
                );
            }
            "quit" => {
                self.stop_search();
                return false;
            }
            _ => {}
        }
        true
    }

    /// The static evaluation the `eval` command prints: mop-up if it applies, then the network if enabled and loaded, otherwise the classical evaluation.
    fn static_eval(&self) -> i32 {
        if let Some(score) = bare_king_eval(&self.pos) {
            return score;
        }
        if self.params.nnue {
            if let Some(net) = nnue::current() {
                let mut acc = nnue::Accumulator::default();
                nnue::refresh(&net, &mut acc, &self.pos);
                return nnue::evaluate(&net, &acc, self.pos.side_to_move());
            }
        }
        evaluate(&self.pos)
    }

    /// Prints the identification lines, the option list and `uciok`.
    fn uci(&self) {
        println!("id name Wustite {}", env!("CARGO_PKG_VERSION"));
        println!("id author Taheem");
        println!("option name Hash type spin default {DEFAULT_HASH_MB} min 1 max {MAX_HASH_MB}");
        println!("option name Threads type spin default 1 min 1 max {MAX_THREADS}");
        println!("option name Move Overhead type spin default 30 min 0 max 5000");
        println!("option name Ponder type check default false");
        println!(
            "option name UseNNUE type check default {}",
            self.params.nnue
        );
        println!("option name EvalFile type string default <internal>");
        println!("option name Clear Hash type button");
        println!("uciok");
    }

    /// Handles `position`: parses `startpos` or `fen`, then plays each listed move, recording every position hash for repetition detection.
    fn position(&mut self, tokens: &[&str]) {
        let (mut pos, index) = match tokens.first() {
            Some(&"startpos") => (Position::startpos(), 1),
            Some(&"fen") => {
                let end = tokens
                    .iter()
                    .position(|&t| t == "moves")
                    .unwrap_or(tokens.len());
                match Position::from_fen(&tokens[1..end].join(" ")) {
                    Ok(pos) => (pos, end),
                    Err(error) => {
                        println!("info string {error}");
                        return;
                    }
                }
            }
            _ => return,
        };

        let mut hist = vec![pos.hash()];
        if tokens.get(index) == Some(&"moves") {
            for text in &tokens[index + 1..] {
                match parse_move(&pos, text) {
                    Some(mv) => {
                        pos = pos.make_move(mv);
                        hist.push(pos.hash());
                    }
                    None => {
                        println!("info string illegal move {text}");
                        break;
                    }
                }
            }
        }
        self.pos = pos;
        self.game_hist = hist;
    }

    /// Handles `go`: `go perft` runs in place; any other `go` computes the limits,
    /// spawns the search worker and returns immediately so that `stop` and
    /// `ponderhit` can be read while it runs.
    fn go(&mut self, tokens: &[&str]) {
        self.join_worker();
        let go = GoParams::parse(tokens);

        if let Some(depth) = go.perft {
            let start = Instant::now();
            let rows = divide(&self.pos, depth);
            let mut total = 0u64;
            for (mv, nodes) in &rows {
                println!("{mv}: {nodes}");
                total += nodes;
            }
            let millis = start.elapsed().as_millis().max(1) as u64;
            println!("\nNodes searched: {total}");
            println!("info string perft {depth} time {millis} ms");
            return;
        }

        self.shared.stop.store(false, Ordering::Relaxed);
        self.shared.nodes.store(0, Ordering::Relaxed);
        self.shared.ponder_offset_ms.store(0, Ordering::Relaxed);
        self.shared.pondering.store(go.ponder, Ordering::Relaxed);
        self.tt.new_search();

        self.search_start = Instant::now();
        let limits = build_limits(
            &go,
            self.pos.side_to_move(),
            self.overhead_ms,
            self.search_start,
        );
        self.infinite = limits.infinite || go.ponder;

        let mut heuristics = std::mem::take(&mut self.heuristics);
        heuristics.resize_with(self.threads, Heuristics::new);
        heuristics.truncate(self.threads);

        let pos = self.pos;
        let hist = self.game_hist.clone();
        let tt = Arc::clone(&self.tt);
        let shared = Arc::clone(&self.shared);
        let params = self.params;

        let spawned = std::thread::Builder::new()
            .stack_size(WORKER_STACK_BYTES)
            .spawn(move || {
                let mut heuristics = heuristics;
                let result = think(
                    &pos,
                    &hist,
                    &limits,
                    &tt,
                    &shared,
                    params,
                    &mut heuristics,
                    true,
                );
                if result.best_move.is_none() {
                    println!("bestmove 0000");
                } else if result.ponder_move.is_none() {
                    println!("bestmove {}", result.best_move);
                } else {
                    println!(
                        "bestmove {} ponder {}",
                        result.best_move, result.ponder_move
                    );
                }
                heuristics
            });

        match spawned {
            Ok(handle) => self.worker = Some(handle),
            Err(error) => {
                println!("info string could not start search thread: {error}");
                self.heuristics = vec![Heuristics::new()];
            }
        }
    }

    /// Handles `setoption`. Recognised UCI options are applied directly; any other name is passed, with spaces removed, to [`Params::set`](crate::search::Params::set) when the value is a number.
    fn setoption(&mut self, tokens: &[&str]) {
        let value_at = tokens.iter().position(|&t| t.eq_ignore_ascii_case("value"));
        let name_end = value_at.unwrap_or(tokens.len());
        let name_start = usize::from(
            tokens
                .first()
                .is_some_and(|t| t.eq_ignore_ascii_case("name")),
        );
        let name = tokens[name_start.min(name_end)..name_end]
            .join(" ")
            .to_ascii_lowercase();
        let value = value_at
            .map(|i| tokens[i + 1..].join(" "))
            .unwrap_or_default();

        match name.as_str() {
            "hash" => {
                if let Ok(mb) = value.parse::<usize>() {
                    self.stop_search();
                    self.tt = Arc::new(TranspositionTable::new(mb.clamp(1, MAX_HASH_MB)));
                }
            }
            "threads" => {
                if let Ok(count) = value.parse::<usize>() {
                    self.threads = count.clamp(1, MAX_THREADS);
                }
            }
            "move overhead" => {
                if let Ok(ms) = value.parse::<u64>() {
                    self.overhead_ms = ms.min(5000);
                }
            }
            "ponder" => {}
            "usennue" => {
                let on = matches!(value.to_ascii_lowercase().as_str(), "true" | "1");
                self.stop_search();
                self.params.nnue = on;
                self.tt.clear();
            }
            "evalfile" => {
                self.stop_search();
                let result = if value.is_empty() || value == "<internal>" {
                    nnue::load_embedded()
                } else {
                    nnue::load_file(&value)
                };
                match result {
                    Ok(()) => self.tt.clear(),
                    Err(error) => println!("info string could not load network: {error}"),
                }
            }
            "clear hash" => {
                self.stop_search();
                self.tt.clear();
            }
            other => {
                let compact: String = other.chars().filter(|c| !c.is_whitespace()).collect();
                if let Ok(number) = value.parse::<i32>() {
                    self.params.set(&compact, number);
                }
            }
        }
    }
}

/// Runs the UCI loop on standard input and output until `quit` or end of input.
///
/// This is what `wustite` does with no arguments. On end of input an infinite or
/// pondering search is stopped so the process can exit cleanly.
///
/// ```no_run
/// wustite::uci::run();
/// ```
pub fn run() {
    let mut engine = Engine::new();
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else {
            break;
        };
        if !engine.handle(&line) {
            return;
        }
    }
    if engine.infinite {
        engine.shared.stop.store(true, Ordering::Relaxed);
        engine.shared.pondering.store(false, Ordering::Relaxed);
    }
    engine.join_worker();
}
