//! # Wustite
//!
//! A UCI chess engine written in Rust with zero external crates, using an NNUE
//! network trained entirely with the tools in this repository.
//!
//! The crate builds both the `wustite` binary and a library you can call
//! directly for move generation, perft, static evaluation, or search.
//!
//! ## How a move flows through the engine
//!
//! The UCI thread owns stdin and never searches directly. When `position` comes
//! in, it applies the move list with `make_move` and keeps every position hash
//! for repetition checks. When `go` arrives, it turns the clock settings into
//! [`search::SearchLimits`] and spawns a worker thread (with a 64 MiB stack)
//! running [`search::think`].
//!
//! Thread 0 drives iterative deepening and prints `info` lines; any Lazy SMP
//! helpers search the same root with staggered depth skips. The threads only
//! share the lockless transposition table and a few atomics, so `stop` and
//! `ponderhit` are picked up immediately without waiting on a mutex.
//!
//! ## Running the binary
//!
//! Point a UCI GUI like Cute Chess, Arena, or En Croissant at the executable:
//!
//! ```text
//! cargo install wustite
//! wustite
//! ```
//!
//! See [`uci`] for all supported commands and options. The binary also has CLI
//! subcommands: `wustite bench`, `wustite perft`, `wustite divide`, and
//! `wustite suite`.
//!
//! ## Using the library
//!
//! [`Position`] uses copy-make, so playing moves never mutates the parent and
//! there's no unmake state to manage:
//!
//! ```
//! use wustite::Position;
//! use wustite::movegen::{legal_moves, parse_move};
//! use wustite::perft::perft;
//! use wustite::search::search_depth;
//!
//! let pos = Position::startpos();
//! assert_eq!(legal_moves(&pos).len(), 20);
//! assert_eq!(perft(&pos, 3), 8_902);
//!
//! let after_e4 = pos.make_move(parse_move(&pos, "e2e4").unwrap());
//! assert_eq!(
//!     after_e4.to_fen(),
//!     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
//! );
//!
//! let result = search_depth(&after_e4, 4, 8);
//! println!("best reply: {}", result.best_move);
//! ```
//!
//! For full control over limits, threads, and search flags, use [`search::think`].
//!
//! ## Modules
//!
//! | Module | What it does |
//! | --- | --- |
//! | [`types`] | [`Color`], [`PieceType`], [`Piece`], [`Square`], and the 16-bit [`Move`] |
//! | [`bitboard`] | `u64` square set [`Bitboard`] with shifts and bit iteration |
//! | [`attacks`] | Compile-time attack tables (magic bitboards or BMI2 PEXT) and ray geometry |
//! | [`magic_numbers`] | Precomputed rook and bishop magic multipliers |
//! | [`zobrist`] | Compile-time Zobrist hashing keys |
//! | [`board`] | Copy-make [`Position`], incremental hash, and FEN parsing |
//! | [`movegen`] | Legal-only staged move generation into a fixed [`MoveList`] |
//! | [`perft`] | Perft, `divide`, and the reference test suite |
//! | [`eval`] | Classical PeSTO evaluation and bare-king mop-up |
//! | [`see`] | Static exchange evaluation |
//! | [`nnue`] | NNUE inference, incremental accumulators, SIMD kernels, and file I/O |
//! | [`tt`] | Lockless shared transposition table |
//! | [`search`] | PVS alpha-beta, iterative deepening, pruning, move ordering, Lazy SMP |
//! | [`time`] | UCI `go` parsing and soft/hard time budgets |
//! | [`uci`] | UCI protocol front end |
//! | [`bench`](mod@bench) | Fixed-depth search benchmark |
//!
//! ## Build flags
//!
//! * `cargo build --release` gives a portable build (magic bitboards, runtime AVX2 on x86-64, scalar elsewhere).
//! * `RUSTFLAGS="-C target-cpu=native"` turns on BMI2 `pext` and AVX-512BW (needs Rust 1.89+) where supported.
//! * On AMD Zen 1 and Zen 2, `pext` is microcoded and slower than magic multiplication, so add `-C target-feature=-bmi2`.
//!
//! The crate is pre-1.0, so public APIs may still change between minor versions.

#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

pub mod attacks;
pub mod bench;
pub mod bitboard;
pub mod board;
pub mod eval;
pub mod magic_numbers;
pub mod movegen;
pub mod nnue;
pub mod perft;
pub mod search;
pub mod see;
pub mod time;
pub mod tt;
pub mod types;
pub mod uci;
pub mod zobrist;

pub use bitboard::Bitboard;
pub use board::Position;
pub use movegen::MoveList;
pub use types::{Color, Move, Piece, PieceType, Square};
