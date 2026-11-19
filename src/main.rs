use std::env;
use std::process::ExitCode;
use std::time::Instant;

use wustite::Position;
use wustite::bench;
use wustite::perft::{SUITE, divide, perft};
use wustite::search::Params;
use wustite::uci;

fn usage() -> ExitCode {
    eprintln!("usage:");
    eprintln!("  wustite                       start the UCI engine");
    eprintln!("  wustite bench [depth]         fixed-depth search benchmark");
    eprintln!("  wustite perft <depth> [fen]");
    eprintln!("  wustite divide <depth> [fen]");
    eprintln!("  wustite suite [max-depth]");
    ExitCode::from(2)
}

fn run_bench(args: &[String]) -> ExitCode {
    let depth = args.first().and_then(|d| d.parse().ok()).unwrap_or(12);
    let report = bench::run(depth, Params::default(), true);
    println!();
    println!("nodes {}", report.nodes);
    println!("time {:.3}s", report.seconds);
    println!("nps {}", nodes_per_second(report.nodes, report.seconds));
    ExitCode::SUCCESS
}

fn position_from_args(args: &[String]) -> Result<Position, String> {
    if args.is_empty() {
        Ok(Position::startpos())
    } else {
        Position::from_fen(&args.join(" ")).map_err(|e| e.to_string())
    }
}

fn nodes_per_second(nodes: u64, seconds: f64) -> u64 {
    if seconds > 0.0 {
        (nodes as f64 / seconds) as u64
    } else {
        0
    }
}

fn run_perft(args: &[String], show_divide: bool) -> ExitCode {
    let Some(depth) = args.first().and_then(|d| d.parse::<u32>().ok()) else {
        return usage();
    };
    let pos = match position_from_args(&args[1..]) {
        Ok(pos) => pos,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };

    let start = Instant::now();
    let total = if show_divide {
        let rows = divide(&pos, depth);
        for (mv, nodes) in &rows {
            println!("{mv}: {nodes}");
        }
        rows.iter().map(|(_, n)| n).sum()
    } else {
        perft(&pos, depth)
    };
    let seconds = start.elapsed().as_secs_f64();

    println!();
    println!("nodes {total}");
    println!("time {:.3}s", seconds);
    println!("nps {}", nodes_per_second(total, seconds));
    ExitCode::SUCCESS
}

fn run_suite(args: &[String]) -> ExitCode {
    let max_depth = args
        .first()
        .and_then(|d| d.parse::<u32>().ok())
        .unwrap_or(5);
    let mut failures = 0;
    let mut total_nodes = 0u64;
    let start = Instant::now();

    for case in SUITE {
        let pos = match Position::from_fen(case.fen) {
            Ok(pos) => pos,
            Err(e) => {
                eprintln!("{}: {e}", case.name);
                failures += 1;
                continue;
            }
        };
        for &(depth, expected) in case.results {
            if depth > max_depth {
                continue;
            }
            let found = perft(&pos, depth);
            total_nodes += found;
            let verdict = if found == expected { "ok" } else { "FAIL" };
            if found != expected {
                failures += 1;
            }
            println!(
                "{:<22} depth {depth} {found:>12} (expected {expected:>12}) {verdict}",
                case.name
            );
        }
    }

    let seconds = start.elapsed().as_secs_f64();
    println!();
    println!(
        "{total_nodes} nodes in {seconds:.2}s, {} nps",
        nodes_per_second(total_nodes, seconds)
    );
    if failures == 0 {
        println!("all perft checks passed");
        ExitCode::SUCCESS
    } else {
        println!("{failures} perft checks failed");
        ExitCode::FAILURE
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("perft") => run_perft(&args[1..], false),
        Some("divide") => run_perft(&args[1..], true),
        Some("suite") => run_suite(&args[1..]),
        Some("bench") => run_bench(&args[1..]),
        None => {
            uci::run();
            ExitCode::SUCCESS
        }
        _ => usage(),
    }
}
