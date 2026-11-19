use wustite::Position;
use wustite::perft::{SUITE, perft};

const RANDOM_POSITIONS: &str = include_str!("data/random_positions.txt");

fn node_limit() -> u64 {
    if cfg!(debug_assertions) {
        1_000_000
    } else {
        200_000_000
    }
}

#[test]
fn standard_suite() {
    for case in SUITE {
        let pos = Position::from_fen(case.fen).expect("suite FEN must parse");
        for &(depth, expected) in case.results {
            if expected > node_limit() {
                continue;
            }
            assert_eq!(
                perft(&pos, depth),
                expected,
                "{} at depth {depth}",
                case.name
            );
        }
    }
}

#[test]
fn random_positions_match_python_chess() {
    let mut checked = 0;
    for line in RANDOM_POSITIONS.lines() {
        let mut parts = line.split(';');
        let fen = parts.next().expect("fen column");
        let expected: Vec<u64> = parts.map(|p| p.parse().expect("node count")).collect();
        let pos = Position::from_fen(fen).unwrap_or_else(|e| panic!("{fen}: {e}"));
        for (i, &nodes) in expected.iter().enumerate() {
            let depth = i as u32 + 1;
            assert_eq!(perft(&pos, depth), nodes, "{fen} at depth {depth}");
        }
        checked += 1;
    }
    assert!(checked >= 400, "fixture shrank to {checked} positions");
}
