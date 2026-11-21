//! Board consistency, staged movegen parity, hashing, and FEN validation tests.
//!
//! Walks 3-ply move trees across the perft suite and random positions, verifying
//! `Position::is_consistent`, king safety, and `GEN_TACTICAL + GEN_QUIET == GEN_ALL`.

use wustite::movegen::{GEN_ALL, GEN_QUIET, GEN_TACTICAL, MoveList, generate, parse_move};
use wustite::perft::SUITE;
use wustite::{Move, Position};

const RANDOM_POSITIONS: &str = include_str!("data/random_positions.txt");

fn all_fens() -> Vec<&'static str> {
    let mut fens: Vec<&'static str> = SUITE.iter().map(|c| c.fen).collect();
    fens.extend(
        RANDOM_POSITIONS
            .lines()
            .take(120)
            .filter_map(|l| l.split(';').next()),
    );
    fens
}

fn walk(pos: &Position, depth: u32) {
    assert!(
        pos.is_consistent(),
        "inconsistent position {}",
        pos.to_fen()
    );

    let mut all = MoveList::new();
    let mut tactical = MoveList::new();
    let mut quiet = MoveList::new();
    generate::<GEN_ALL>(pos, &mut all);
    generate::<GEN_TACTICAL>(pos, &mut tactical);
    generate::<GEN_QUIET>(pos, &mut quiet);

    let mut staged: Vec<Move> = tactical.iter().chain(quiet.iter()).copied().collect();
    let mut expected: Vec<Move> = all.iter().copied().collect();
    staged.sort_by_key(|m| m.raw());
    expected.sort_by_key(|m| m.raw());
    assert_eq!(
        staged,
        expected,
        "staged generation differs from full generation in {}",
        pos.to_fen()
    );

    for &mv in all.iter() {
        let next = pos.make_move(mv);
        let mover = pos.side_to_move();
        assert!(
            !next.attacked_by(next.king_square(mover), !mover, next.occupied()),
            "{mv} leaves the king in check in {}",
            pos.to_fen()
        );
        if depth > 1 {
            walk(&next, depth - 1);
        } else {
            assert!(
                next.is_consistent(),
                "{mv} corrupts the position {}",
                pos.to_fen()
            );
        }
    }
}

#[test]
fn move_trees_stay_consistent() {
    for fen in all_fens() {
        let pos = Position::from_fen(fen).expect("fen");
        walk(&pos, 3);
    }
}

#[test]
fn fen_round_trip_preserves_position() {
    for fen in all_fens() {
        let pos = Position::from_fen(fen).expect("fen");
        let again = Position::from_fen(&pos.to_fen()).expect("regenerated fen");
        assert_eq!(pos, again, "{fen}");
        assert_eq!(pos.hash(), again.hash());
    }
    assert_eq!(
        Position::startpos().to_fen(),
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    );
}

#[test]
fn transpositions_share_a_hash() {
    let start = Position::startpos();
    let play = |pos: &Position, moves: &[&str]| {
        let mut current = *pos;
        for text in moves {
            let mv = parse_move(&current, text).unwrap_or_else(|| panic!("illegal {text}"));
            current = current.make_move(mv);
        }
        current
    };
    let a = play(&start, &["g1f3", "g8f6", "b1c3", "b8c6"]);
    let b = play(&start, &["b1c3", "b8c6", "g1f3", "g8f6"]);
    assert_eq!(a.hash(), b.hash());
    assert_eq!(a.hash(), a.compute_hash());
    assert_ne!(a.hash(), start.hash());
}

#[test]
fn null_move_updates_hash_and_side() {
    let pos = Position::startpos();
    let after = pos.make_null_move();
    assert_eq!(after.side_to_move(), !pos.side_to_move());
    assert_eq!(after.hash(), after.compute_hash());
    assert_eq!(after.make_null_move().hash(), pos.hash());
}

#[test]
fn rejects_malformed_fens() {
    let bad = [
        "",
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",
        "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "p3k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - e5 0 1",
        "4k3/8/8/8/8/8/4q3/4K3 b - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - x 1",
    ];
    for fen in bad {
        assert!(Position::from_fen(fen).is_err(), "accepted {fen:?}");
    }
}

#[test]
fn accepts_lenient_but_sound_fens() {
    let good = [
        "4k3/8/8/8/8/8/8/4K2R b K - 0 1",
        "4k2r/8/8/8/8/8/8/4K3 w k",
        "4k3/8/8/8/8/8/8/4K3 w",
    ];
    for fen in good {
        assert!(Position::from_fen(fen).is_ok(), "rejected {fen:?}");
    }
}
