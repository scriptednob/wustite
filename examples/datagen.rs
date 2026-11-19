use std::env;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use wustite::movegen::legal_moves;
use wustite::search::{Heuristics, Params, SearchLimits, SharedState, think};
use wustite::tt::TranspositionTable;
use wustite::{Color, Position};

const RANDOM_PLIES: [usize; 2] = [8, 9];
const OPENING_LIMIT: i32 = 450;
const SCORE_LIMIT: i32 = 3000;
const ADJUDICATE_SCORE: i32 = 1500;
const ADJUDICATE_PLIES: u32 = 4;
const MAX_GAME_PLIES: usize = 300;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

struct Settings {
    out: String,
    threads: usize,
    positions: u64,
    nodes: u64,
    seed: u64,
    params: Params,
}

struct Record {
    fen: String,
    score: i32,
    stm: Color,
}

struct Worker {
    tt: TranspositionTable,
    heuristics: Vec<Box<Heuristics>>,
    nodes: u64,
    params: Params,
}

impl Worker {
    fn search(&mut self, pos: &Position, hist: &[u64]) -> (wustite::Move, i32) {
        self.tt.new_search();
        let shared = SharedState::new();
        let mut limits = SearchLimits::depth_only(64);
        limits.nodes = self.nodes;
        let result = think(
            pos,
            hist,
            &limits,
            &self.tt,
            &shared,
            self.params,
            &mut self.heuristics,
            false,
        );
        (result.best_move, result.score)
    }

    fn play(&mut self, rng: &mut Rng) -> Option<Vec<(Record, u8)>> {
        self.tt.clear();
        for heur in &mut self.heuristics {
            heur.clear();
        }

        let mut pos = Position::startpos();
        let mut hist = vec![pos.hash()];
        let plies = RANDOM_PLIES[rng.below(RANDOM_PLIES.len())];
        for _ in 0..plies {
            let moves = legal_moves(&pos);
            if moves.is_empty() {
                return None;
            }
            pos = pos.make_move(moves[rng.below(moves.len())]);
            hist.push(pos.hash());
        }
        if legal_moves(&pos).is_empty() {
            return None;
        }
        let (_, opening_score) = self.search(&pos, &hist);
        if opening_score.abs() > OPENING_LIMIT {
            return None;
        }

        let mut records: Vec<Record> = Vec::new();
        let mut winning_streak = 0u32;
        let mut streak_side = Color::White;
        let outcome: u8;

        loop {
            let ply = hist.len() - 1;
            let repeats = hist.iter().filter(|&&h| h == pos.hash()).count();
            if pos.halfmove_clock() >= 100
                || repeats >= 3
                || pos.insufficient_material()
                || ply >= MAX_GAME_PLIES
            {
                outcome = 1;
                break;
            }

            let (best, score) = self.search(&pos, &hist);
            if best.is_none() {
                outcome = if pos.in_check() {
                    if pos.side_to_move() == Color::White {
                        0
                    } else {
                        2
                    }
                } else {
                    1
                };
                break;
            }

            let side = pos.side_to_move();
            let white_score = if side == Color::White { score } else { -score };
            if white_score.abs() >= ADJUDICATE_SCORE {
                let leader = if white_score > 0 {
                    Color::White
                } else {
                    Color::Black
                };
                if winning_streak > 0 && leader == streak_side {
                    winning_streak += 1;
                } else {
                    winning_streak = 1;
                    streak_side = leader;
                }
                if winning_streak >= ADJUDICATE_PLIES {
                    outcome = if leader == Color::White { 2 } else { 0 };
                    break;
                }
            } else {
                winning_streak = 0;
            }

            if !pos.in_check()
                && !best.is_capture()
                && !best.is_promotion()
                && score.abs() < SCORE_LIMIT
            {
                records.push(Record {
                    fen: pos.to_fen(),
                    score,
                    stm: side,
                });
            }

            pos = pos.make_move(best);
            hist.push(pos.hash());
        }

        Some(
            records
                .into_iter()
                .map(|record| {
                    let wdl = match (outcome, record.stm) {
                        (1, _) => 1,
                        (2, Color::White) | (0, Color::Black) => 2,
                        _ => 0,
                    };
                    (record, wdl)
                })
                .collect(),
        )
    }
}

fn parse_settings() -> Result<Settings, String> {
    let mut settings = Settings {
        out: String::from("data.txt"),
        threads: 1,
        positions: 1_000_000,
        nodes: 6000,
        seed: 1,
        params: Params {
            nnue: false,
            ..Params::default()
        },
    };
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {}", args[i]))?;
        match args[i].as_str() {
            "--out" => settings.out = value.clone(),
            "--threads" => settings.threads = value.parse().map_err(|_| "bad --threads")?,
            "--positions" => settings.positions = value.parse().map_err(|_| "bad --positions")?,
            "--nodes" => settings.nodes = value.parse().map_err(|_| "bad --nodes")?,
            "--seed" => settings.seed = value.parse().map_err(|_| "bad --seed")?,
            "--net" => {
                wustite::nnue::load_file(value).map_err(|error| format!("bad --net: {error}"))?;
                settings.params.nnue = true;
            }
            other => return Err(format!("unknown option {other}")),
        }
        i += 2;
    }
    settings.threads = settings.threads.max(1);
    Ok(settings)
}

fn main() -> ExitCode {
    let settings = match parse_settings() {
        Ok(settings) => settings,
        Err(message) => {
            eprintln!("{message}");
            eprintln!(
                "usage: datagen [--out FILE] [--threads N] [--positions N] [--nodes N] [--seed N] [--net FILE]"
            );
            return ExitCode::from(2);
        }
    };

    let file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&settings.out)
    {
        Ok(file) => file,
        Err(error) => {
            eprintln!("cannot open {}: {error}", settings.out);
            return ExitCode::from(2);
        }
    };
    let writer = Arc::new(Mutex::new(BufWriter::new(file)));
    let written = Arc::new(AtomicU64::new(0));
    let games = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    std::thread::scope(|scope| {
        for t in 0..settings.threads {
            let writer = Arc::clone(&writer);
            let written = Arc::clone(&written);
            let games = Arc::clone(&games);
            let settings = &settings;
            scope.spawn(move || {
                let mut rng = Rng::new(settings.seed.wrapping_add(t as u64 * 7919 + 1));
                let mut worker = Worker {
                    tt: TranspositionTable::new(16),
                    heuristics: vec![Heuristics::new()],
                    nodes: settings.nodes,
                    params: settings.params,
                };
                while written.load(Ordering::Relaxed) < settings.positions {
                    let Some(records) = worker.play(&mut rng) else {
                        continue;
                    };
                    if records.is_empty() {
                        continue;
                    }
                    let mut text = String::new();
                    for (record, wdl) in &records {
                        text.push_str(&format!("{};{};{}\n", record.fen, record.score, wdl));
                    }
                    if let Ok(mut out) = writer.lock() {
                        let _ = out.write_all(text.as_bytes());
                        let _ = out.flush();
                    }
                    written.fetch_add(records.len() as u64, Ordering::Relaxed);
                    games.fetch_add(1, Ordering::Relaxed);
                }
            });
        }

        let written = Arc::clone(&written);
        let games = Arc::clone(&games);
        let target = settings.positions;
        scope.spawn(move || {
            let mut ticks = 0u32;
            while written.load(Ordering::Relaxed) < target {
                std::thread::sleep(Duration::from_secs(1));
                ticks += 1;
                if ticks % 60 != 0 {
                    continue;
                }
                let n = written.load(Ordering::Relaxed);
                let secs = start.elapsed().as_secs_f64().max(1.0);
                eprintln!(
                    "{} positions, {} games, {:.0} positions/s",
                    n,
                    games.load(Ordering::Relaxed),
                    n as f64 / secs
                );
            }
        });
    });

    if let Ok(mut out) = writer.lock() {
        let _ = out.flush();
    }
    eprintln!(
        "done: {} positions in {:.0}s",
        written.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );
    ExitCode::SUCCESS
}
