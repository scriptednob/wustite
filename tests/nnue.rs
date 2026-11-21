//! NNUE correctness tests: incremental vs. full refresh, SIMD vs. scalar parity,
//! perspective symmetry, feature bijection, and `WNNU` file validation.

use wustite::Position;
use wustite::movegen::legal_moves;
use wustite::nnue::{
    Accumulator, FILE_BYTES, HEADER_BYTES, HIDDEN, INPUTS, LoadError, MAGIC, Network, VERSION,
    evaluate, feature_index, refresh, scalar_evaluate, scalar_refresh, serialize, update,
};
use wustite::perft::SUITE;
use wustite::types::{Color, Piece};

const RANDOM_POSITIONS: &str = include_str!("data/random_positions.txt");

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn range(&mut self, low: i32, high: i32) -> i32 {
        low + (self.next() % (high - low + 1) as u64) as i32
    }
}

fn random_network(seed: u64) -> Network {
    let mut rng = Rng(seed | 1);
    let ft_weights: Vec<i16> = (0..INPUTS * HIDDEN)
        .map(|_| rng.range(-45, 45) as i16)
        .collect();
    let ft_bias: Vec<i16> = (0..HIDDEN).map(|_| rng.range(-30, 60) as i16).collect();
    let out_weights: Vec<i16> = (0..2 * HIDDEN)
        .map(|_| rng.range(-127, 127) as i16)
        .collect();
    let bytes = serialize(&ft_weights, &ft_bias, &out_weights, rng.range(-5000, 5000));
    Network::from_bytes(&bytes).expect("valid network")
}

fn assert_same(a: &Accumulator, b: &Accumulator, context: &str) {
    assert_eq!(a.v[0], b.v[0], "white perspective differs: {context}");
    assert_eq!(a.v[1], b.v[1], "black perspective differs: {context}");
}

#[test]
fn every_move_updates_incrementally_like_a_refresh() {
    let net = random_network(7);
    let mut fens: Vec<String> = SUITE.iter().map(|c| c.fen.to_string()).collect();
    fens.extend(
        RANDOM_POSITIONS
            .lines()
            .take(150)
            .filter_map(|l| l.split(';').next())
            .map(str::to_string),
    );
    fens.push("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1".to_string());
    fens.push("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1".to_string());
    fens.push("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3".to_string());
    fens.push("1n2k3/P7/8/8/8/8/p7/1N2K3 w - - 0 1".to_string());
    fens.push("1n2k3/P7/8/8/8/8/p7/1N2K3 b - - 0 1".to_string());

    let mut checked = 0;
    for fen in &fens {
        let pos = Position::from_fen(fen).expect("fen");
        let mut base = Accumulator::default();
        refresh(&net, &mut base, &pos);
        let mut reference = Accumulator::default();
        scalar_refresh(&net, &mut reference, &pos);
        assert_same(&base, &reference, fen);

        for &mv in legal_moves(&pos).iter() {
            let child = pos.make_move(mv);
            let mut incremental = Accumulator::default();
            update(&net, &base, &mut incremental, &pos, mv);
            let mut fresh = Accumulator::default();
            scalar_refresh(&net, &mut fresh, &child);
            assert_same(&incremental, &fresh, &format!("{fen} move {mv}"));
            checked += 1;
        }
    }
    assert!(checked > 2000, "only {checked} moves checked");
}

#[test]
fn long_random_games_stay_in_sync() {
    let net = random_network(11);
    let mut rng = Rng(0xABCDEF);
    for fen in RANDOM_POSITIONS
        .lines()
        .take(40)
        .filter_map(|l| l.split(';').next())
    {
        let mut pos = Position::from_fen(fen).expect("fen");
        let mut acc = Accumulator::default();
        refresh(&net, &mut acc, &pos);
        for _ in 0..60 {
            let moves = legal_moves(&pos);
            if moves.is_empty() {
                break;
            }
            let mv = moves[(rng.next() % moves.len() as u64) as usize];
            let mut next_acc = Accumulator::default();
            update(&net, &acc, &mut next_acc, &pos, mv);
            pos = pos.make_move(mv);
            acc = next_acc;
            let mut fresh = Accumulator::default();
            scalar_refresh(&net, &mut fresh, &pos);
            assert_same(&acc, &fresh, fen);
        }
    }
}

#[test]
fn simd_and_scalar_evaluations_agree() {
    for seed in [3u64, 5, 9] {
        let net = random_network(seed);
        for fen in RANDOM_POSITIONS
            .lines()
            .take(150)
            .filter_map(|l| l.split(';').next())
        {
            let pos = Position::from_fen(fen).expect("fen");
            let mut acc = Accumulator::default();
            refresh(&net, &mut acc, &pos);
            for stm in Color::ALL {
                assert_eq!(
                    evaluate(&net, &acc, stm),
                    scalar_evaluate(&net, &acc, stm),
                    "{fen}"
                );
            }
        }
    }
}

#[test]
fn feature_indices_are_a_bijection_per_perspective() {
    for perspective in Color::ALL {
        let mut seen = vec![false; INPUTS];
        for piece in Piece::ALL {
            for sq in 0..64 {
                let index = feature_index(perspective, piece, wustite::Square::from_idx(sq));
                assert!(index < INPUTS);
                assert!(!seen[index], "duplicate index {index}");
                seen[index] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }
}

#[test]
fn the_two_perspectives_mirror_each_other() {
    let net = random_network(21);
    let fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
    let mirrored = "rnbqk2r/pppp1ppp/5n2/2b1p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 4 4";
    let a = Position::from_fen(fen).expect("fen");
    let b = Position::from_fen(mirrored).expect("fen");
    let mut acc_a = Accumulator::default();
    let mut acc_b = Accumulator::default();
    refresh(&net, &mut acc_a, &a);
    refresh(&net, &mut acc_b, &b);
    assert_eq!(acc_a.v[0], acc_b.v[1]);
    assert_eq!(acc_a.v[1], acc_b.v[0]);
    assert_eq!(
        evaluate(&net, &acc_a, Color::White),
        evaluate(&net, &acc_b, Color::Black)
    );
}

#[test]
fn network_files_round_trip_and_reject_bad_input() {
    let net = random_network(13);
    let bytes = net.to_bytes();
    assert_eq!(bytes.len(), FILE_BYTES);
    let again = Network::from_bytes(&bytes).expect("round trip");
    assert_eq!(again.to_bytes(), bytes);

    let mut bad_magic = bytes.clone();
    bad_magic[0] = b'X';
    assert!(matches!(
        Network::from_bytes(&bad_magic),
        Err(LoadError::Magic)
    ));

    let mut bad_version = bytes.clone();
    bad_version[4] = 99;
    assert!(matches!(
        Network::from_bytes(&bad_version),
        Err(LoadError::Version(99))
    ));

    let mut bad_shape = bytes.clone();
    bad_shape[12] = 7;
    assert!(matches!(
        Network::from_bytes(&bad_shape),
        Err(LoadError::Shape { .. })
    ));

    assert!(matches!(
        Network::from_bytes(&bytes[..bytes.len() - 2]),
        Err(LoadError::Size { .. })
    ));
    assert!(matches!(
        Network::from_bytes(&bytes[..HEADER_BYTES - 1]),
        Err(LoadError::Size { .. })
    ));

    let mut loud = bytes.clone();
    let offset = FILE_BYTES - 4 - 2;
    loud[offset..offset + 2].copy_from_slice(&200i16.to_le_bytes());
    assert!(matches!(
        Network::from_bytes(&loud),
        Err(LoadError::OutputRange)
    ));
    assert_eq!(&bytes[0..4], &MAGIC);
    assert_eq!(
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        VERSION
    );
}
