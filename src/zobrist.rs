//! Zobrist hashing keys for positions.
//!
//! A Zobrist hash identifies a position by XOR-ing together a random 64-bit key
//! for each feature that is present: every (piece, square) pair, the castling
//! rights, the en passant file, and a key that is included when black is to move.
//! XOR is its own inverse, so the hash can be updated incrementally when a move
//! is made by XOR-ing out the features that disappear and XOR-ing in the ones that
//! appear, without looking at the rest of the board. [`crate::board::Position`]
//! does exactly that in `make_move`, and [`crate::board::Position::compute_hash`]
//! recomputes it from scratch as a cross-check.
//!
//! The key table is generated at compile time with the SplitMix64 generator from
//! a fixed seed. That has two practical consequences: the keys are identical in
//! every build on every platform, so hashes are reproducible and the same
//! position always has the same key in test output, and there is no run-time
//! start-up cost.
//!
//! Collisions are possible in principle (two positions sharing a key) but with
//! 64-bit keys they are astronomically rare at the table sizes engines use, and
//! the transposition table additionally validates entries with its own check word.
//!
//! ```
//! use wustite::Position;
//!
//! let start = Position::startpos();
//! assert_eq!(start.hash(), start.compute_hash());
//! ```

/// One step of the SplitMix64 generator. Takes the generator state and returns
/// the next state together with a well-mixed 64-bit output.
const fn splitmix64(state: u64) -> (u64, u64) {
    let next = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = next;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (next, z ^ (z >> 31))
}

/// The complete set of Zobrist keys. A single instance, [`ZOBRIST`], is built at compile time.
pub struct Zobrist {
    /// One key for each of the 12 [`crate::types::Piece`] values on each of the 64 squares, indexed `[piece][square]`.
    pub piece_square: [[u64; 64]; 12],
    /// One key per castling-rights bitmask value `0..16` (see [`crate::board::ALL_CASTLING`]). The whole mask is hashed as a unit.
    pub castling: [u64; 16],
    /// One key per file, XOR-ed in while an en passant capture is actually available on that file.
    pub ep_file: [u64; 8],
    /// XOR-ed in when black is to move, so the same arrangement with a different side to move hashes differently.
    pub side: u64,
}

impl Zobrist {
    /// Fills every key from the fixed-seed SplitMix64 stream, in a fixed order, entirely at compile time.
    const fn build() -> Zobrist {
        let mut state = 0x5755_5354_4954_4521u64;
        let mut keys = Zobrist {
            piece_square: [[0; 64]; 12],
            castling: [0; 16],
            ep_file: [0; 8],
            side: 0,
        };

        let mut piece = 0;
        while piece < 12 {
            let mut sq = 0;
            while sq < 64 {
                let (s, v) = splitmix64(state);
                state = s;
                keys.piece_square[piece][sq] = v;
                sq += 1;
            }
            piece += 1;
        }

        let mut i = 0;
        while i < 16 {
            let (s, v) = splitmix64(state);
            state = s;
            keys.castling[i] = v;
            i += 1;
        }

        let mut i = 0;
        while i < 8 {
            let (s, v) = splitmix64(state);
            state = s;
            keys.ep_file[i] = v;
            i += 1;
        }

        let (_, v) = splitmix64(state);
        keys.side = v;
        keys
    }
}

/// The global key table.
pub static ZOBRIST: Zobrist = Zobrist::build();
