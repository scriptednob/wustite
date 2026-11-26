//! Universal Chess Interface (UCI) front end.
//!
//! [`run`] reads UCI commands from stdin and prints responses to stdout. It is
//! what the `wustite` binary runs when invoked with no arguments.
//!
//! # Threading model
//!
//! The main UCI thread only reads lines and flips atomic flags—it never runs a
//! search itself. On `go`, it spawns a search worker (with a 64 MiB stack to
//! hold the searcher's ply arrays) calling [`think`], which in turn spawns any
//! Lazy SMP helper threads. That keeps `stop` and `ponderhit` instant even
//! mid-search. Move-ordering heuristics are moved into the worker at `go` and
//! returned when the worker is joined so history carries over between moves.
//!
//! # Supported commands
//!
//! | Command | What it does |
//! | --- | --- |
//! | `uci` | Print `id` lines, options, and `uciok` |
//! | `isready` | Print `readyok` |
//! | `ucinewgame` | Stop any search, clear TT and heuristics, reset to `startpos` |
//! | `position startpos [moves ...]` | Load `startpos` and play moves, keeping hashes for repetition checks |
//! | `position fen <fen> [moves ...]` | Load a FEN and play moves |
//! | `go ...` | Start searching (or run `go perft <n>`) |
//! | `stop` | Abort the active search and emit `bestmove` |
//! | `ponderhit` | Opponent played the ponder move; start the real clock from now |
//! | `setoption name <n> [value <v>]` | Change a standard or hidden tuning option |
//! | `quit` | Stop searching and exit |
//!
//! Non-standard debug helpers also work in UCI mode: `d` (print board/FEN/key),
//! `eval` (static eval), `bench [depth]`, and `go perft <n>`. Unknown commands
//! are ignored per the UCI spec, and EOF acts like `quit`.
//!
//! # Options
//!
//! | Name | Type | Default | Range | Description |
//! | --- | --- | --- | --- | --- |
//! | `Hash` | spin | `16` | `1..=65536` | TT size in MiB (reallocates on change) |
//! | `Threads` | spin | `1` | `1..=256` | Lazy SMP worker count |
//! | `Move Overhead` | spin | `30` | `0..=5000` | Safety buffer in ms deducted per move |
//! | `Ponder` | check | `false` | | Advertises pondering support to the GUI |
//! | `UseNNUE` | check | `true` | | Toggle between NNUE and classical PeSTO |
//! | `EvalFile` | string | `<internal>` | | Path to a `.nnue` file, or `<internal>` |
//! | `Clear Hash` | button | | | Wipe the transposition table |
//!
//! Option names are case-insensitive. Every search flag and margin in [`Params`]
//! can also be set via `setoption` with an integer (`0` = off, non-zero = on),
//! e.g. `setoption name NMP value 0`. Normal builds don't list them in `uci`.
//! Builds with the `tune` Cargo feature do, because match runners such as
//! fastchess only send options that the engine lists.
//!
//! # Example session
//!
//! ```text
//! > uci
//! < id name Wustite (these brackets represent latest version number because updating this is tedious)
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

/// Search worker stack size (needed for the searcher's large fixed-size ply arrays).
const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;
/// Default transposition table size in MiB.
const DEFAULT_HASH_MB: usize = 16;
/// Upper bound for the `Hash` option in MiB.
const MAX_HASH_MB: usize = 65_536;
/// Upper bound for the `Threads` option.
const MAX_THREADS: usize = 256;

/// Persistent UCI session state (board, game hashes, TT, atomics, options, and active worker handle).
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
    /// Initialize an engine at `startpos` with default UCI options.
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

    /// Wait for the search worker to exit and reclaim its heuristics (recreating them if it panicked).
    fn join_worker(&mut self) {
        if let Some(handle) = self.worker.take() {
            match handle.join() {
                Ok(heuristics) => self.heuristics = heuristics,
                Err(_) => self.heuristics = vec![Heuristics::new()],
            }
        }
    }

    /// Signal the active search to stop, clear pondering, and join the worker.
    fn stop_search(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.pondering.store(false, Ordering::Relaxed);
        self.join_worker();
    }

    /// Credit elapsed ponder time into `ponder_offset_ms` and hand control to the normal clock.
    fn ponder_hit(&mut self) {
        if self.shared.pondering.load(Ordering::Relaxed) {
            let offset = self.search_start.elapsed().as_millis() as u64;
            self.shared
                .ponder_offset_ms
                .store(offset, Ordering::Relaxed);
            self.shared.pondering.store(false, Ordering::Relaxed);
        }
    }

    /// Dispatch one UCI input line; returns `false` on `quit`.
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

    /// Compute static eval for the `eval` debug command (mop-up -> NNUE -> PeSTO).
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

    /// Print engine identity, declared options, and `uciok`.
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
        if cfg!(feature = "tune") {
            for line in tune_options(&self.params) {
                println!("{line}");
            }
        }
        println!("uciok");
    }

    /// Parse `position startpos|fen ... [moves ...]` and record position hashes for repetition checks.
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

    /// Handle `go perft` inline, or spawn a background search thread for normal/ponder searches.
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

    /// Apply standard UCI options directly, forwarding unknown integer options to [`Params::set`].
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

/// Run the UCI stdin/stdout loop until `quit` or EOF.
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

fn tune_options(p: &Params) -> Vec<String> {
    let flags = [
        ("aspiration", p.aspiration),
        ("lmr", p.lmr),
        ("nmp", p.nmp),
        ("rfp", p.rfp),
        ("razoring", p.razoring),
        ("futility", p.futility),
        ("lmp", p.lmp),
        ("seepruning", p.see_pruning),
        ("iir", p.iir),
        ("capturehistory", p.capture_history),
        ("conthistory", p.cont_history),
        ("singular", p.singular),
        ("probcut", p.probcut),
        ("tteval", p.tt_eval),
        ("histlmr", p.hist_lmr),
        ("histpruning", p.hist_pruning),
    ];
    let numbers = [
        ("histprunemargin", p.hist_prune_margin),
        ("rfpmargin", p.rfp_margin),
        ("razormargin", p.razor_margin),
        ("futilitybase", p.futility_base),
        ("futilityslope", p.futility_slope),
        ("nmpbase", p.nmp_base),
        ("nmpverifydepth", p.nmp_verify_depth),
        ("seecapturemargin", p.see_capture_margin),
        ("seequietmargin", p.see_quiet_margin),
        ("aspirationdelta", p.aspiration_delta),
        ("probcutmargin", p.probcut_margin),
        ("singulardepth", p.singular_depth),
    ];
    let spin = |name: &str, value: i32, max: i32| {
        format!("option name {name} type spin default {value} min 0 max {max}")
    };
    flags
        .iter()
        .map(|&(name, on)| spin(name, i32::from(on), 1))
        .chain(
            numbers
                .iter()
                .map(|&(name, value)| spin(name, value, 100_000)),
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tune_option_is_accepted() {
        let lines = tune_options(&Params::default());
        assert_eq!(lines.len(), 28);
        for line in lines {
            let name = line.split_whitespace().nth(2).unwrap();
            assert!(Params::default().set(name, 1), "{name}");
        }
    }
}
