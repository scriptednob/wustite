//! # Wustite
//!
//! A UCI chess engine written in Rust with no dependencies, using an NNUE
//! evaluation that is trained entirely with the tools in this repository.
//!
//! This crate is both the engine library and, through the `wustite` binary, the
//! engine itself. The documentation here is meant to be read as a guide to how a
//! modern alpha-beta engine is built: every module explains *what* it does and
//! *why*, and the sections below show how the pieces fit together.
//!
//! ## One move, end to end
//!
//! ```text
//! GUI --"position startpos moves e2e4 e7e5"--> UCI thread    (uci)
//!                                                |  Position::from_fen, make_move
//!                                                |  game history: hash of every position
//! GUI --"go wtime 60000 btime 60000 ..."------> UCI thread
//!                                                |  GoParams::parse, build_limits  (time)
//!                                                v
//!                                         worker thread, 64 MiB stack
//!                                                |  think()                        (search)
//!                  +-----------------------------+--------------------+
//!                  v                                                  v
//!          main searcher (id 0)                          helper searchers (id 1..N)
//!          iterative deepening, prints `info`            same root, own stacks,
//!                  |                                     skip some depths
//!                  +------------ shared TranspositionTable ------------+   (tt)
//!                                                |
//!                                                v
//!                                    "bestmove e2e4 ponder e7e5"
//! ```
//!
//! The UCI thread only parses text and flips atomic flags, so `stop` and
//! `ponderhit` are handled at once even in the middle of a long search. Searchers
//! share nothing but the transposition table and a few atomics, which keeps the
//! whole engine free of locks.
//!
//! ## Using the engine
//!
//! Install it and point a UCI-capable chess GUI such as Cute Chess, Arena or En
//! Croissant at the executable:
//!
//! ```text
//! cargo install wustite
//! wustite
//! ```
//!
//! It then speaks the Universal Chess Interface on standard input and output.
//! See [`uci`] for every command and option. The binary also has a few
//! command-line tools: `wustite bench`, `wustite perft`, `wustite divide` and
//! `wustite suite`.
//!
//! ## Using the library
//!
//! Positions are immutable values that are cloned (copy-make) rather than
//! mutated, so generating moves, playing them and searching are all plain
//! function calls:
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
//! Searching with full control over limits, threads and options is done with
//! [`search::think`]; see the [`search`] module for an example.
//!
//! ## Module map
//!
//! The modules form a stack. Each depends only on those above it in the table.
//!
//! | Module | Role |
//! | --- | --- |
//! | [`types`] | [`Color`], [`PieceType`], [`Piece`], [`Square`] and the 16-bit [`Move`] |
//! | [`bitboard`] | [`Bitboard`], a set of squares in a `u64`, with shifts and iteration |
//! | [`attacks`] | Attack tables for every piece: magic bitboards or PEXT, built at compile time |
//! | [`zobrist`] | Compile-time hashing keys |
//! | [`board`] | [`Position`]: copy-make, incremental hash, FEN |
//! | [`movegen`] | Fully legal move generation in staged modes, into a fixed-size [`MoveList`] |
//! | [`perft`] | Move-generation correctness test and the reference suite |
//! | [`eval`] | Classical PeSTO evaluation and the bare-king mop-up rule |
//! | [`see`] | Static exchange evaluation |
//! | [`nnue`] | The neural-network evaluation: inference, incremental accumulators, file format |
//! | [`tt`] | The lockless shared transposition table |
//! | [`search`] | Alpha-beta (PVS), iterative deepening, pruning, move ordering, Lazy SMP |
//! | [`time`] | `go` parsing and time management |
//! | [`uci`] | The protocol front end |
//! | [`bench`](mod@bench) | Fixed-depth benchmark |
//! | [`magic_numbers`] | The constants behind the magic bitboards |
//!
//! ## Design principles
//!
//! * **No dependencies.** The crate depends only on `std`.
//! * **No allocation in the search.** Move lists, the ply stack, principal
//!   variations and accumulators live on the stack or in preallocated arrays.
//!   Heap memory is used for the transposition table and the history tables, once.
//! * **Compile-time tables.** Attack tables, Zobrist keys, evaluation tables and
//!   the late-move-reduction table are all produced by `const fn` evaluation, so
//!   there is no initialisation step and no start-up cost.
//! * **Legal move generation.** Generated moves never need to be filtered.
//! * **Copy-make.** A 192-byte, cache-aligned [`Position`] is copied for every
//!   move; there is no unmake.
//! * **Everything is tested against a reference.** Move generation against
//!   known node counts and python-chess, incremental network updates against full
//!   recomputation, SIMD against scalar code, and search changes against a build
//!   without them in an SPRT match.
//!
//! ## Build options
//!
//! | Build | Effect |
//! | --- | --- |
//! | `cargo build --release` | Portable: magic bitboards, AVX2 chosen at run time for the network |
//! | `RUSTFLAGS="-C target-cpu=native"` | Enables PEXT lookups (BMI2) and, where available, AVX-512 network code |
//! | `RUSTFLAGS="-C target-feature=+bmi2"` | PEXT lookups only |
//! | non-x86-64 targets | Magic bitboards and scalar network code |
//!
//! PEXT is slow on AMD Zen 1 and Zen 2; build those with
//! `-C target-feature=-bmi2`. The AVX-512 path needs Rust 1.89 or newer.
//!
//! ## Stability
//!
//! The public API exists so the engine's parts can be tested, benchmarked and
//! studied. It is not yet considered stable, and may change between minor
//! versions while the crate is below 1.0.

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
