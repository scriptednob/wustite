//! Compile-time Zobrist hashing keys.
//!
//! A position hash XORs together a 64-bit random key for every piece on its
//! square, the castling rights mask, the active en passant file (if any), and a
//! side-to-move key when black is to move. Since XOR cancels itself out,
//! [`crate::board::Position::make_move`] updates the hash incrementally instead
//! of rescanning the board, and [`crate::board::Position::compute_hash`] stays
//! around to cross-check it in tests.
//!
//! Keys are built at compile time with SplitMix64 from a fixed seed, so hashes
//! are identical across builds and platforms with zero startup cost.
//!
//! ```
//! use wustite::Position;
//!
//! let start = Position::startpos();
//! assert_eq!(start.hash(), start.compute_hash());
//! ```

/// Advance SplitMix64 by one step, returning `(next_state, mixed_output)`.
const fn splitmix64(state: u64) -> (u64, u64) {
    let next = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = next;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (next, z ^ (z >> 31))
}

/// Complete set of Zobrist keys, built at compile time in [`ZOBRIST`].
pub struct Zobrist {
    /// Keys for each [`crate::types::Piece`] on each square, indexed `[piece][square]`.
    pub piece_square: [[u64; 64]; 12],
    /// One key per 4-bit castling rights mask (`0..16`, see [`crate::board::ALL_CASTLING`]).
    pub castling: [u64; 16],
    /// One key per file, mixed in only when an en passant capture is actually available.
    pub ep_file: [u64; 8],
    /// Mixed in whenever black is to move.
    pub side: u64,
}

impl Zobrist {
    /// Fill all keys at compile time from a fixed SplitMix64 seed.
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

/// Global compile-time Zobrist key table.
pub static ZOBRIST: Zobrist = Zobrist::build();
