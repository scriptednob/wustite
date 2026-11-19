use std::time::{Duration, Instant};

use wustite::Position;
use wustite::eval::evaluate;
use wustite::movegen::legal_moves;
use wustite::search::{
    Heuristics, MATE, Params, SearchLimits, SharedState, format_score, search_depth, think,
};
use wustite::see::see_ge;
use wustite::tt::{Bound, TranspositionTable};
use wustite::types::{Move, Square, flag};

const MATES: &str = include_str!("data/mates.txt");
const RANDOM_POSITIONS: &str = include_str!("data/random_positions.txt");

fn random_fens(count: usize) -> Vec<&'static str> {
    RANDOM_POSITIONS
        .lines()
        .filter_map(|line| line.split(';').next())
        .take(count)
        .collect()
}

fn mirror_fen(fen: &str) -> String {
    let mut parts = fen.split_whitespace();
    let placement = parts.next().expect("placement");
    let side = parts.next().expect("side");
    let castling = parts.next().unwrap_or("-");
    let ep = parts.next().unwrap_or("-");
    let swap_case = |c: char| {
        if c.is_ascii_uppercase() {
            c.to_ascii_lowercase()
        } else {
            c.to_ascii_uppercase()
        }
    };
    let ranks: Vec<String> = placement
        .split('/')
        .rev()
        .map(|rank| rank.chars().map(swap_case).collect())
        .collect();
    let mirrored_ep = if ep == "-" {
        "-".to_string()
    } else {
        let mut chars = ep.chars();
        let file = chars.next().expect("ep file");
        let rank = chars.next().expect("ep rank").to_digit(10).expect("digit");
        format!("{file}{}", 9 - rank)
    };
    format!(
        "{} {} {} {mirrored_ep} 0 1",
        ranks.join("/"),
        if side == "w" { "b" } else { "w" },
        castling.chars().map(swap_case).collect::<String>()
    )
}

#[test]
fn finds_forced_mates_with_exact_distance() {
    for line in MATES.lines() {
        let mut parts = line.split(';');
        let fen = parts.next().expect("fen");
        let moves: u32 = parts.next().expect("n").parse().expect("number");
        let solutions: Vec<&str> = parts.next().expect("solutions").split(' ').collect();
        let pos = Position::from_fen(fen).expect("valid fen");
        let result = search_depth(&pos, 9, 16);
        assert_eq!(
            result.score,
            MATE - (2 * moves as i32 - 1),
            "{fen}: expected mate in {moves}, got {}",
            format_score(result.score)
        );
        assert!(
            solutions.contains(&result.best_move.to_string().as_str()),
            "{fen}: {} is not a mating first move",
            result.best_move
        );
    }
}

fn search_with(pos: &Position, depth: i32, params: Params) -> wustite::search::SearchResult {
    let tt = TranspositionTable::new(16);
    let shared = SharedState::new();
    let limits = SearchLimits::depth_only(depth);
    let mut heuristics = vec![Heuristics::new()];
    think(
        pos,
        &[pos.hash()],
        &limits,
        &tt,
        &shared,
        params,
        &mut heuristics,
        false,
    )
}

fn every_feature() -> Params {
    Params {
        razoring: true,
        iir: true,
        capture_history: true,
        cont_history: true,
        singular: true,
        probcut: true,
        tt_eval: true,
        hist_lmr: true,
        hist_pruning: true,
        ..Params::default()
    }
}

#[test]
fn every_feature_keeps_exact_mate_distances() {
    for line in MATES.lines() {
        let mut parts = line.split(';');
        let fen = parts.next().expect("fen");
        let moves: u32 = parts.next().expect("n").parse().expect("number");
        let solutions: Vec<&str> = parts.next().expect("solutions").split(' ').collect();
        let pos = Position::from_fen(fen).expect("valid fen");
        let result = search_with(&pos, 10, every_feature());
        assert_eq!(
            result.score,
            MATE - (2 * moves as i32 - 1),
            "{fen}: expected mate in {moves}, got {}",
            format_score(result.score)
        );
        assert!(
            solutions.contains(&result.best_move.to_string().as_str()),
            "{fen}: {} is not a mating first move",
            result.best_move
        );
    }
}

#[test]
fn every_feature_returns_legal_moves_with_playable_pvs() {
    for fen in random_fens(40) {
        let pos = Position::from_fen(fen).expect("fen");
        let result = search_with(&pos, 7, every_feature());
        let legal = legal_moves(&pos);
        assert!(
            legal.contains(&result.best_move),
            "{fen}: {} is illegal",
            result.best_move
        );
        let mut current = pos;
        for mv in &result.pv {
            assert!(
                legal_moves(&current).contains(mv),
                "{fen}: illegal pv move {mv}"
            );
            current = current.make_move(*mv);
        }
    }
}

#[test]
fn best_moves_are_legal_and_pv_is_playable() {
    for fen in random_fens(25) {
        let pos = Position::from_fen(fen).expect("fen");
        let result = search_depth(&pos, 5, 16);
        let legal = legal_moves(&pos);
        assert!(
            legal.contains(&result.best_move),
            "{fen}: {} is illegal",
            result.best_move
        );
        let mut current = pos;
        for mv in &result.pv {
            assert!(
                legal_moves(&current).contains(mv),
                "{fen}: illegal pv move {mv}"
            );
            current = current.make_move(*mv);
        }
    }
}

#[test]
fn single_threaded_search_is_deterministic() {
    let pos =
        Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
            .expect("fen");
    let first = search_depth(&pos, 7, 16);
    let second = search_depth(&pos, 7, 16);
    assert_eq!(first.nodes, second.nodes);
    assert_eq!(first.best_move, second.best_move);
    assert_eq!(first.score, second.score);
}

#[test]
fn bare_kings_and_lone_minor_are_draws() {
    for fen in [
        "8/8/4k3/8/8/3K4/8/8 w - - 0 1",
        "8/8/4k3/8/8/3K1B2/8/8 w - - 0 1",
    ] {
        let pos = Position::from_fen(fen).expect("fen");
        assert_eq!(search_depth(&pos, 6, 16).score, 0, "{fen}");
    }
}

#[test]
fn stalemate_and_checkmate_positions_have_no_move() {
    for fen in [
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1",
    ] {
        let pos = Position::from_fen(fen).expect("fen");
        let result = search_depth(&pos, 4, 16);
        assert!(result.best_move.is_none(), "{fen}");
    }
}

#[test]
fn respects_time_limits() {
    let pos = Position::startpos();
    let tt = TranspositionTable::new(16);
    let shared = SharedState::new();
    let mut limits = SearchLimits::depth_only(100);
    limits.soft = Some(Duration::from_millis(150));
    limits.hard = Some(Duration::from_millis(150));
    limits.start = Instant::now();
    let mut heuristics = vec![Heuristics::new()];
    let result = think(
        &pos,
        &[pos.hash()],
        &limits,
        &tt,
        &shared,
        Params::default(),
        &mut heuristics,
        false,
    );
    let elapsed = limits.start.elapsed();
    assert!(elapsed < Duration::from_millis(400), "took {elapsed:?}");
    assert!(legal_moves(&pos).contains(&result.best_move));
}

#[test]
fn respects_node_limits() {
    let pos = Position::startpos();
    let tt = TranspositionTable::new(16);
    let shared = SharedState::new();
    let mut limits = SearchLimits::depth_only(100);
    limits.nodes = 50_000;
    let mut heuristics = vec![Heuristics::new()];
    let result = think(
        &pos,
        &[pos.hash()],
        &limits,
        &tt,
        &shared,
        Params::default(),
        &mut heuristics,
        false,
    );
    assert!(
        result.nodes >= 50_000 && result.nodes < 80_000,
        "{}",
        result.nodes
    );
}

#[test]
fn multi_threaded_search_returns_legal_moves() {
    let pos =
        Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
            .expect("fen");
    let tt = TranspositionTable::new(16);
    let shared = SharedState::new();
    let limits = SearchLimits::depth_only(8);
    let mut heuristics: Vec<_> = (0..3).map(|_| Heuristics::new()).collect();
    let result = think(
        &pos,
        &[pos.hash()],
        &limits,
        &tt,
        &shared,
        Params::default(),
        &mut heuristics,
        false,
    );
    assert!(legal_moves(&pos).contains(&result.best_move));
    assert!(result.nodes > 0);
}

#[test]
fn evaluation_is_color_symmetric() {
    for fen in random_fens(150) {
        let pos = Position::from_fen(fen).expect("fen");
        let mirrored = Position::from_fen(&mirror_fen(fen)).expect("mirrored fen");
        assert_eq!(evaluate(&pos), evaluate(&mirrored), "{fen}");
    }
}

#[test]
fn evaluation_orientation_is_sane() {
    let eval = |fen: &str| evaluate(&Position::from_fen(fen).expect("fen"));
    let advanced = eval("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1");
    let home = eval("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    assert!(
        advanced > home,
        "advanced pawn {advanced} vs home pawn {home}"
    );
    let centre = eval("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1");
    let corner = eval("4k3/8/8/8/8/8/8/N3K3 w - - 0 1");
    assert!(
        centre > corner,
        "centralised knight {centre} vs corner knight {corner}"
    );
    let up_queen = eval("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
    assert!(up_queen > 800);
}

#[test]
fn static_exchange_values() {
    let capture = |from: Square, to: Square| Move::new(from, to, flag::CAPTURE);

    let free_pawn = Position::from_fen("4k3/8/8/4p3/8/8/4R3/4K3 w - - 0 1").expect("fen");
    let mv = capture(Square::E2, Square::E5);
    assert!(see_ge(&free_pawn, mv, 100));
    assert!(!see_ge(&free_pawn, mv, 101));

    let defended = Position::from_fen("4k3/8/3p4/4p3/8/8/4R3/4K3 w - - 0 1").expect("fen");
    assert!(see_ge(&defended, mv, -400));
    assert!(!see_ge(&defended, mv, -399));
    assert!(!see_ge(&defended, mv, 0));

    let even = Position::from_fen("4k3/8/3p4/4p3/3P4/8/8/4K3 w - - 0 1").expect("fen");
    let pawn_trade = capture(Square::D4, Square::E5);
    assert!(see_ge(&even, pawn_trade, 0));
    assert!(!see_ge(&even, pawn_trade, 1));
}

#[test]
fn static_exchange_respects_pins() {
    let pos = Position::from_fen("4k3/8/4n3/2p5/8/B7/8/4R1K1 w - - 0 1").expect("fen");
    let mv = Move::new(Square::A3, Square::C5, flag::CAPTURE);
    assert!(see_ge(&pos, mv, 100));
    assert!(!see_ge(&pos, mv, 101));

    let unpinned = Position::from_fen("4k3/8/4n3/2p5/8/B7/8/6K1 w - - 0 1").expect("fen");
    assert!(!see_ge(&unpinned, mv, 0));
    assert!(see_ge(&unpinned, mv, -230));
}

#[test]
fn transposition_table_round_trip() {
    let tt = TranspositionTable::new(1);
    let key = 0x1234_5678_9ABC_DEF0u64;
    assert!(tt.probe(key).is_none());
    let mv = Move::new(Square::E2, Square::E4, flag::DOUBLE_PUSH);
    tt.store(key, mv, -123, 45, 7, Bound::Exact);
    let entry = tt.probe(key).expect("stored entry");
    assert_eq!(entry.mv, mv);
    assert_eq!(entry.score, -123);
    assert_eq!(entry.eval, 45);
    assert_eq!(entry.depth, 7);
    assert_eq!(entry.bound, Bound::Exact);
    assert!(tt.probe(key ^ 1).is_none());
    tt.clear();
    assert!(tt.probe(key).is_none());
}

#[test]
fn transposition_table_survives_concurrent_access() {
    let tt = TranspositionTable::new(1);
    let derive = |key: u64| {
        let score = (key & 0x7FFF) as i32 - 16_000;
        let eval = ((key >> 16) & 0x7FFF) as i32 - 16_000;
        let depth = ((key >> 32) & 0x7F) as i32;
        let mv = Move::from_raw((key >> 40) as u16);
        (mv, score, eval, depth)
    };
    std::thread::scope(|scope| {
        for t in 0..4u64 {
            let tt = &tt;
            scope.spawn(move || {
                let mut state =
                    0x9E37_79B9_7F4A_7C15u64 ^ (t + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                for _ in 0..300_000 {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let key = state % 50_000 + 1;
                    let key = key.wrapping_mul(0x2545_F491_4F6C_DD1D);
                    let (mv, score, eval, depth) = derive(key);
                    if state & 1 == 0 {
                        tt.store(key, mv, score, eval, depth, Bound::Lower);
                    } else if let Some(entry) = tt.probe(key) {
                        assert_eq!(entry.score, score);
                        assert_eq!(entry.eval, eval);
                        assert_eq!(entry.depth, depth);
                        assert_eq!(entry.mv, mv);
                    }
                }
            });
        }
    });
}

fn network_test_bytes() -> Vec<u8> {
    let mut state = 0x0123_4567_89AB_CDEF_u64;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let ft_weights: Vec<i16> = (0..wustite::nnue::INPUTS * wustite::nnue::HIDDEN)
        .map(|_| (next() % 61) as i16 - 30)
        .collect();
    let ft_bias: Vec<i16> = (0..wustite::nnue::HIDDEN)
        .map(|_| (next() % 41) as i16 - 10)
        .collect();
    let out_weights: Vec<i16> = (0..2 * wustite::nnue::HIDDEN)
        .map(|_| (next() % 41) as i16 - 20)
        .collect();
    wustite::nnue::serialize(&ft_weights, &ft_bias, &out_weights, 0)
}

#[test]
fn nnue_search_is_deterministic_legal_and_finds_mates() {
    let net = wustite::nnue::Network::from_bytes(&network_test_bytes()).expect("network");
    wustite::nnue::set_current(Some(std::sync::Arc::new(net)));
    let params = Params {
        nnue: true,
        ..Params::default()
    };

    let pos =
        Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
            .expect("fen");
    let first = search_with(&pos, 7, params);
    let second = search_with(&pos, 7, params);
    assert_eq!(first.nodes, second.nodes);
    assert_eq!(first.best_move, second.best_move);
    assert!(legal_moves(&pos).contains(&first.best_move));

    for line in MATES.lines().take(4) {
        let mut parts = line.split(';');
        let fen = parts.next().expect("fen");
        let moves: u32 = parts.next().expect("n").parse().expect("number");
        let pos = Position::from_fen(fen).expect("valid fen");
        let result = search_with(&pos, 10, params);
        assert_eq!(result.score, MATE - (2 * moves as i32 - 1), "{fen}");
    }

    for fen in random_fens(25) {
        let pos = Position::from_fen(fen).expect("fen");
        let result = search_with(&pos, 6, params);
        assert!(legal_moves(&pos).contains(&result.best_move), "{fen}");
    }

    let aggressive = Params {
        nnue: true,
        singular: true,
        singular_depth: 3,
        probcut: true,
        razoring: true,
        hist_pruning: true,
        hist_lmr: true,
        capture_history: true,
        ..Params::default()
    };
    for fen in random_fens(6) {
        let pos = Position::from_fen(fen).expect("fen");
        let result = search_with(&pos, 9, aggressive);
        assert!(legal_moves(&pos).contains(&result.best_move), "{fen}");
    }
    wustite::nnue::load_embedded().expect("embedded network");
}

#[test]
fn bare_king_eval_applies_only_to_lone_king_endings() {
    use wustite::eval::bare_king_eval;

    let applies = [
        "8/8/8/8/8/8/k2K4/5Q2 w - - 0 1",
        "7k/8/8/6K1/6R1/8/8/8 b - - 0 1",
    ];
    for fen in applies {
        let pos = Position::from_fen(fen).expect("fen");
        let score = bare_king_eval(&pos).expect("applies");
        let mirrored = Position::from_fen(&mirror_fen(fen)).expect("fen");
        assert_eq!(bare_king_eval(&mirrored), Some(score), "{fen}");
    }

    let strong = Position::from_fen("8/8/8/8/8/8/k2K4/5Q2 w - - 0 1").expect("fen");
    let weak = Position::from_fen("8/8/8/8/8/8/k2K4/5Q2 b - - 0 1").expect("fen");
    assert!(bare_king_eval(&strong).expect("applies") > 800);
    assert!(bare_king_eval(&weak).expect("applies") < -800);

    let cornered = Position::from_fen("k7/8/8/8/8/8/8/KQ6 w - - 0 1").expect("fen");
    let central = Position::from_fen("8/8/8/3k4/8/8/8/KQ6 w - - 0 1").expect("fen");
    assert!(bare_king_eval(&cornered) > bare_king_eval(&central));

    let none = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/8/8/8/8/8/k2K4/5N2 w - - 0 1",
        "8/8/8/8/8/8/kp1K4/5Q2 w - - 0 1",
        "8/8/8/8/8/5P2/k2K4/8 w - - 0 1",
    ];
    for fen in none {
        let pos = Position::from_fen(fen).expect("fen");
        assert_eq!(bare_king_eval(&pos), None, "{fen}");
    }
}
