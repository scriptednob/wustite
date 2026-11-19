//! NNUE: an efficiently updatable neural network evaluation.
//!
//! NNUE ("Efficiently Updatable Neural Network", from the Japanese shogi
//! community, ported to chess by Stockfish) is a small neural network whose first
//! layer is so sparse and so incrementally updatable that it can run inside a
//! search at millions of positions per second on a CPU. Wustite uses the
//! simplest useful architecture and trains it entirely with the tools in this
//! repository (`examples/datagen.rs` and `examples/train.rs`).
//!
//! # Architecture
//!
//! ```text
//!  768 inputs (perspective A)          768 inputs (perspective B)
//!         |  sparse, <= 32 active           |
//!         v                                 v
//!  [feature transformer: 768 -> 128]  [same weights, same bias]
//!         |                                 |
//!    accumulator A (128 x i16)         accumulator B (128 x i16)
//!         |  SCReLU                         |  SCReLU
//!         v                                 v
//!    w_us . x^2   +   w_them . x^2   -> one output -> centipawns
//! ```
//!
//! * **Inputs.** For each of the two perspectives there are 768 binary inputs:
//!   one for each (own or enemy, piece type, square) combination,
//!   `2 colours x 6 piece types x 64 squares`. A position has at most 32 pieces,
//!   so at most 32 inputs are active. See [`feature_index`].
//! * **Perspectives.** The same weights are used twice: once with the board seen
//!   from white's side, once from black's. "Seen from black's side" means the
//!   ranks are flipped (`square ^ 56`) and "own" and "enemy" are swapped. The
//!   accumulator for the side to move is paired with the output weights for
//!   "us" and the other accumulator with the weights for "them", so the network
//!   knows whose turn it is.
//! * **Hidden layer.** 128 units ([`HIDDEN`]) with a `SCReLU` activation:
//!   `clamp(x, 0, QA)^2`. The squared clipped ReLU is a little more expressive
//!   than a plain clipped ReLU and costs one extra multiply in integer
//!   arithmetic.
//! * **Output.** One value: the dot product of the activated hidden units with
//!   the output weights, rescaled to centipawns.
//!
//! # Incremental updates
//!
//! The feature transformer is a plain sum:
//! `accumulator = bias + sum of the weight columns of the active features`.
//! When a move is made only two to five features change (the moved piece leaves
//! one square and arrives on another, plus a captured piece, plus a rook when
//! castling). So instead of recomputing the sum over all pieces, [`update`]
//! copies the parent's [`Accumulator`] and **adds** the columns of features that
//! appeared and **subtracts** the columns of features that disappeared. That is
//! a handful of 128-element vector additions per move per perspective.
//!
//! The search keeps one accumulator per ply and writes the child's accumulator
//! right before descending into it. A debug build re-derives every accumulator
//! from scratch at each evaluation and asserts equality, and the test suite
//! compares incremental updates with [`refresh`] after every legal move of many
//! positions.
//!
//! # Quantisation
//!
//! All arithmetic is integer, using the usual NNUE scheme:
//!
//! | Constant | Value | Role |
//! | --- | --- | --- |
//! | [`QA`] | 255 | scale of the feature-transformer output (`i16` activations) |
//! | [`QB`] | 64 | scale of the output weights |
//! | [`SCALE`] | 400 | maps the network's output to centipawns |
//! | [`OUTPUT_LIMIT`] | 127 | largest allowed magnitude of an output weight |
//!
//! The final score is
//!
//! ```text
//! sum    = sum over hidden units of clamp(a, 0, QA)^2 * w       (both perspectives)
//! hidden = sum / QA + output_bias
//! score  = hidden * SCALE / (QA * QB)
//! ```
//!
//! Output weights are limited to `+-127` on purpose: with an activation of at
//! most 255, the 16-bit SIMD multiply `x * w` is at most `255 * 127 = 32385`,
//! which fits an `i16` without overflow. [`Network::from_bytes`] rejects files that
//! violate the limit.
//!
//! # SIMD
//!
//! Accumulator updates and the output dot product are vectorised. The
//! implementation is chosen as follows:
//!
//! | Target | Implementation | Chosen |
//! | --- | --- | --- |
//! | x86-64 with AVX-512BW enabled at compile time | AVX-512 | at compile time (needs Rust 1.89+) |
//! | any other x86-64 | AVX2 when the CPU has it, detected once at run time | at run time |
//! | everything else, including ARM | portable scalar code | at compile time |
//!
//! [`simd_description`] reports which is active. The scalar versions
//! ([`scalar_refresh`], [`scalar_evaluate`]) are always compiled and serve as
//! the reference that tests compare the SIMD results against.
//!
//! # The network file
//!
//! Files use the format `WNNU` version 1, little-endian throughout:
//!
//! | Offset | Size | Content |
//! | --- | --- | --- |
//! | 0 | 4 | magic bytes `WNNU` ([`MAGIC`]) |
//! | 4 | 4 | version, `u32` ([`VERSION`]) |
//! | 8 | 4 | number of inputs, `u32` (768) |
//! | 12 | 4 | hidden size, `u32` (128) |
//! | 16 | `2 * 768 * 128` | feature-transformer weights, `i16`, one 128-element column per input feature |
//! | then | `2 * 128` | feature-transformer bias, `i16` |
//! | then | `2 * 256` | output weights, `i16`: 128 for "us" followed by 128 for "them" |
//! | then | 4 | output bias, `i32` |
//!
//! The total is [`FILE_BYTES`] = 197,396 bytes. The header records the shape so
//! that a network of a different size is rejected with a clear error instead of
//! being misread.
//!
//! # The active network
//!
//! A process-wide network slot holds the network every new search uses. It is
//! initialised with the network embedded in the binary (`nets/wustite.nnue`).
//! [`load_file`] replaces it from a file and [`load_embedded`] restores the
//! built-in one, which is what the UCI `EvalFile` option does.
//!
//! # Examples
//!
//! Evaluate a position with the built-in network:
//!
//! ```
//! use wustite::{nnue, Position};
//!
//! let net = nnue::current().expect("the embedded network is valid");
//! let pos = Position::startpos();
//! let mut acc = nnue::Accumulator::default();
//! nnue::refresh(&net, &mut acc, &pos);
//! let score = nnue::evaluate(&net, &acc, pos.side_to_move());
//! assert!(score.abs() < 300);
//! ```
//!
//! Update the accumulator incrementally and check it against a full recompute:
//!
//! ```
//! use wustite::{nnue, Position};
//! use wustite::movegen::parse_move;
//!
//! let net = nnue::current().unwrap();
//! let pos = Position::startpos();
//! let mut parent = nnue::Accumulator::default();
//! nnue::refresh(&net, &mut parent, &pos);
//!
//! let mv = parse_move(&pos, "e2e4").unwrap();
//! let child = pos.make_move(mv);
//!
//! let mut incremental = nnue::Accumulator::default();
//! nnue::update(&net, &parent, &mut incremental, &pos, mv);
//!
//! let mut fresh = nnue::Accumulator::default();
//! nnue::refresh(&net, &mut fresh, &child);
//! assert_eq!(incremental.v, fresh.v);
//! ```

use std::fmt;
use std::sync::{Arc, OnceLock, RwLock};

use crate::board::Position;
use crate::types::{Color, Move, Piece, PieceType, Square, flag};

/// Input features per perspective: 2 colours x 6 piece types x 64 squares = 768.
pub const INPUTS: usize = 768;
/// Width of the hidden layer (the feature transformer's output): 128 units.
pub const HIDDEN: usize = 128;
/// Quantisation scale of the hidden layer. Activations are clamped to `0..=QA` before squaring.
pub const QA: i32 = 255;
/// Quantisation scale of the output weights.
pub const QB: i32 = 64;
/// Factor converting the network's output to centipawns.
pub const SCALE: i32 = 400;
/// Largest permitted magnitude of an output weight, chosen so a 16-bit multiply cannot overflow. Files that exceed it are rejected.
pub const OUTPUT_LIMIT: i16 = 127;
/// The four magic bytes, `WNNU`, that start every network file.
pub const MAGIC: [u8; 4] = *b"WNNU";
/// The network file format version this build understands.
pub const VERSION: u32 = 1;
/// Size of the file header in bytes: magic, version, input count and hidden size.
pub const HEADER_BYTES: usize = 16;
/// Exact size in bytes of a valid network file.
pub const FILE_BYTES: usize = HEADER_BYTES + 2 * (INPUTS * HIDDEN + HIDDEN + 2 * HIDDEN) + 4;

/// The network compiled into the binary from `nets/wustite.nnue`.
static DEFAULT_NET: &[u8] = include_bytes!("../nets/wustite.nnue");

/// Why a network could not be loaded.
///
/// ```
/// use wustite::nnue::{LoadError, Network};
///
/// assert!(matches!(Network::from_bytes(&[0u8; 4]), Err(LoadError::Size { .. })));
/// assert!(matches!(Network::from_bytes(&[0u8; 16]), Err(LoadError::Magic)));
/// ```
#[derive(Debug)]
pub enum LoadError {
    /// The file has the wrong length.
    Size {
        /// The length in bytes of a valid network file, [`FILE_BYTES`].
        expected: usize,
        /// The length in bytes of the data that was supplied.
        found: usize,
    },
    /// The file does not start with the `WNNU` magic bytes.
    Magic,
    /// The file format version is not supported. The payload is the version found.
    Version(u32),
    /// The network was trained with a different input or hidden size than this build supports.
    Shape {
        /// The number of input features recorded in the file header.
        inputs: u32,
        /// The hidden layer width recorded in the file header.
        hidden: u32,
    },
    /// An output weight exceeds [`OUTPUT_LIMIT`], which could overflow the 16-bit SIMD multiply.
    OutputRange,
    /// The file could not be read. The payload is a message that includes the path.
    Io(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Size { expected, found } => {
                write!(f, "network file has {found} bytes, expected {expected}")
            }
            LoadError::Magic => write!(f, "not a Wustite network file"),
            LoadError::Version(v) => write!(f, "unsupported network version {v}"),
            LoadError::Shape { inputs, hidden } => write!(
                f,
                "network shape {inputs}x{hidden} does not match this build ({INPUTS}x{HIDDEN})"
            ),
            LoadError::OutputRange => write!(f, "output weight outside the supported range"),
            LoadError::Io(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// The weights of a loaded network.
///
/// Create one with [`Network::from_bytes`]. Networks are immutable after loading
/// and shared between search threads behind an `Arc`.
pub struct Network {
    ft_weights: Box<[i16]>,
    ft_bias: [i16; HIDDEN],
    out_weights: [i16; 2 * HIDDEN],
    out_bias: i32,
}

/// The feature-transformer output for both perspectives: two arrays of 128
/// `i16` values (index 0 for white's view, 1 for black's).
///
/// The struct is 64-byte aligned so SIMD loads and stores never straddle cache
/// lines. [`update`] derives a child's accumulator from its parent's.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    /// `v[perspective][unit]`, where `perspective` is [`crate::types::Color::idx`].
    pub v: [[i16; HIDDEN]; 2],
}

impl Default for Accumulator {
    fn default() -> Accumulator {
        Accumulator {
            v: [[0; HIDDEN]; 2],
        }
    }
}

/// Reads a little-endian `i16` at a byte offset.
fn read_i16(bytes: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

/// Reads a little-endian `u32` at a byte offset.
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Encodes network parts in the `WNNU` file format.
///
/// Used by `examples/train.rs` to write trained networks. Expects the
/// feature-transformer weights as `INPUTS * HIDDEN` values (one 128-element
/// column per feature, in feature order), the bias as `HIDDEN` values and the
/// output weights as `2 * HIDDEN` values (the "us" half first).
pub fn serialize(
    ft_weights: &[i16],
    ft_bias: &[i16],
    out_weights: &[i16],
    out_bias: i32,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(FILE_BYTES);
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&(INPUTS as u32).to_le_bytes());
    bytes.extend_from_slice(&(HIDDEN as u32).to_le_bytes());
    for &w in ft_weights.iter().chain(ft_bias).chain(out_weights) {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes.extend_from_slice(&out_bias.to_le_bytes());
    bytes
}

impl Network {
    /// Parses a network from file contents.
    ///
    /// # Errors
    ///
    /// Returns a [`LoadError`] if the data is too short, has the wrong magic, version
    /// or shape, has the wrong total length, or contains an output weight beyond
    /// [`OUTPUT_LIMIT`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Network, LoadError> {
        if bytes.len() < HEADER_BYTES {
            return Err(LoadError::Size {
                expected: FILE_BYTES,
                found: bytes.len(),
            });
        }
        if bytes[0..4] != MAGIC {
            return Err(LoadError::Magic);
        }
        let version = read_u32(bytes, 4);
        if version != VERSION {
            return Err(LoadError::Version(version));
        }
        let inputs = read_u32(bytes, 8);
        let hidden = read_u32(bytes, 12);
        if inputs as usize != INPUTS || hidden as usize != HIDDEN {
            return Err(LoadError::Shape { inputs, hidden });
        }
        if bytes.len() != FILE_BYTES {
            return Err(LoadError::Size {
                expected: FILE_BYTES,
                found: bytes.len(),
            });
        }

        let mut offset = HEADER_BYTES;
        let mut ft_weights = vec![0i16; INPUTS * HIDDEN].into_boxed_slice();
        for w in ft_weights.iter_mut() {
            *w = read_i16(bytes, offset);
            offset += 2;
        }
        let mut ft_bias = [0i16; HIDDEN];
        for b in ft_bias.iter_mut() {
            *b = read_i16(bytes, offset);
            offset += 2;
        }
        let mut out_weights = [0i16; 2 * HIDDEN];
        for w in out_weights.iter_mut() {
            *w = read_i16(bytes, offset);
            offset += 2;
            if w.abs() > OUTPUT_LIMIT {
                return Err(LoadError::OutputRange);
            }
        }
        let out_bias = read_u32(bytes, offset) as i32;

        Ok(Network {
            ft_weights,
            ft_bias,
            out_weights,
            out_bias,
        })
    }

    /// Encodes the network in the `WNNU` format. `Network::from_bytes(&net.to_bytes())` reproduces the same weights.
    pub fn to_bytes(&self) -> Vec<u8> {
        serialize(
            &self.ft_weights,
            &self.ft_bias,
            &self.out_weights,
            self.out_bias,
        )
    }

    /// The 128 weights for one input feature.
    #[inline(always)]
    fn column(&self, feature: usize) -> &[i16] {
        &self.ft_weights[feature * HIDDEN..(feature + 1) * HIDDEN]
    }
}

/// The input-feature index (`0..768`) of `piece` standing on `sq`, as seen from
/// `perspective`.
///
/// ```text
/// index = (enemy * 6 + piece_type) * 64 + square
/// ```
///
/// where `enemy` is 1 when the piece's colour differs from `perspective`, and
/// `square` is the square index, vertically flipped (`^ 56`) when the perspective
/// is black. The two perspectives therefore see the same board as mirror images
/// of each other, which lets one set of weights serve both colours.
///
/// ```
/// use wustite::nnue::{feature_index, INPUTS};
/// use wustite::{Color, Piece, Square};
///
/// let white_view = feature_index(Color::White, Piece::WhitePawn, Square::E2);
/// let black_view = feature_index(Color::Black, Piece::BlackPawn, Square::E7);
/// assert_eq!(white_view, black_view);
/// assert!(white_view < INPUTS);
/// ```
#[inline(always)]
pub fn feature_index(perspective: Color, piece: Piece, sq: Square) -> usize {
    let enemy = usize::from(piece.color() != perspective);
    let square = sq.idx() ^ (56 * perspective.idx());
    (enemy * 6 + piece.piece_type().idx()) * 64 + square
}

/// Computes `acc` from scratch for `pos`: bias plus the weight column of every
/// piece on the board, for both perspectives.
///
/// This is the slow path, used to initialise the root of a search and as the
/// reference for incremental updates.
pub fn refresh(net: &Network, acc: &mut Accumulator, pos: &Position) {
    for perspective in Color::ALL {
        let v = &mut acc.v[perspective.idx()];
        *v = net.ft_bias;
        for sq in pos.occupied() {
            if let Some(piece) = pos.piece_on(sq) {
                simd::add_assign(v, net.column(feature_index(perspective, piece, sq)));
            }
        }
    }
}

/// Like [`refresh`] but always uses the portable scalar code. The reference implementation for tests and debug assertions.
pub fn scalar_refresh(net: &Network, acc: &mut Accumulator, pos: &Position) {
    for perspective in Color::ALL {
        let v = &mut acc.v[perspective.idx()];
        *v = net.ft_bias;
        for sq in pos.occupied() {
            if let Some(piece) = pos.piece_on(sq) {
                simd::scalar::add_assign(v, net.column(feature_index(perspective, piece, sq)));
            }
        }
    }
}

/// Computes the accumulator for the position after `mv` from its parent.
///
/// `pos` is the position **before** the move and `src` its accumulator; the
/// result is written to `dst`. At most three features are removed (the moving
/// piece, a captured piece, the castling rook's origin) and at most two are added
/// (the arriving piece, possibly a promoted piece, the castling rook's
/// destination).
///
/// A null move changes no features, so the search just copies the accumulator.
pub fn update(net: &Network, src: &Accumulator, dst: &mut Accumulator, pos: &Position, mv: Move) {
    let us = pos.side_to_move();
    let from = mv.from_sq();
    let to = mv.to_sq();
    let Some(piece) = pos.piece_on(from) else {
        *dst = *src;
        return;
    };

    let mut removed = [(piece, from); 3];
    let mut removed_count = 1;
    let mut added = [(piece, to); 2];
    let mut added_count = 1;

    if let Some(promoted) = mv.promotion_piece() {
        added[0] = (Piece::new(us, promoted), to);
    }
    if mv.is_capture() {
        let captured_sq = if mv.is_en_passant() {
            to.offset(if us == Color::White { -8 } else { 8 })
        } else {
            to
        };
        if let Some(captured) = pos.piece_on(captured_sq) {
            removed[removed_count] = (captured, captured_sq);
            removed_count += 1;
        }
    }
    if mv.flags() == flag::KING_CASTLE {
        let rook = Piece::new(us, PieceType::Rook);
        removed[removed_count] = (rook, to.offset(1));
        removed_count += 1;
        added[added_count] = (rook, to.offset(-1));
        added_count += 1;
    } else if mv.flags() == flag::QUEEN_CASTLE {
        let rook = Piece::new(us, PieceType::Rook);
        removed[removed_count] = (rook, to.offset(-2));
        removed_count += 1;
        added[added_count] = (rook, to.offset(1));
        added_count += 1;
    }

    for perspective in Color::ALL {
        let index = perspective.idx();
        let out = &mut dst.v[index];
        *out = src.v[index];
        for &(added_piece, sq) in &added[..added_count] {
            simd::add_assign(out, net.column(feature_index(perspective, added_piece, sq)));
        }
        for &(removed_piece, sq) in &removed[..removed_count] {
            simd::sub_assign(
                out,
                net.column(feature_index(perspective, removed_piece, sq)),
            );
        }
    }
}

/// Converts the raw dot-product `sum` into centipawns: rescales by `QA`, adds the output bias, then scales by `SCALE / (QA * QB)`.
#[inline(always)]
fn finish(net: &Network, sum: i64) -> i32 {
    let hidden = sum / QA as i64 + net.out_bias as i64;
    (hidden * SCALE as i64 / (QA * QB) as i64) as i32
}

/// Evaluates a position from its accumulator, in centipawns from the point of
/// view of `stm`, the side to move.
///
/// The accumulator of `stm` is combined with the "us" output weights and the
/// other side's with the "them" weights.
pub fn evaluate(net: &Network, acc: &Accumulator, stm: Color) -> i32 {
    let us = &acc.v[stm.idx()];
    let them = &acc.v[stm.flip().idx()];
    let sum = simd::screlu_dot(us, &net.out_weights[..HIDDEN]) as i64
        + simd::screlu_dot(them, &net.out_weights[HIDDEN..]) as i64;
    finish(net, sum)
}

/// Like [`evaluate`] but always uses the portable scalar code. SIMD and scalar results are required to be identical.
pub fn scalar_evaluate(net: &Network, acc: &Accumulator, stm: Color) -> i32 {
    let us = &acc.v[stm.idx()];
    let them = &acc.v[stm.flip().idx()];
    let sum = simd::scalar::screlu_dot(us, &net.out_weights[..HIDDEN]) as i64
        + simd::scalar::screlu_dot(them, &net.out_weights[HIDDEN..]) as i64;
    finish(net, sum)
}

/// The process-wide slot holding the active network. Initialised lazily from the embedded network.
fn slot() -> &'static RwLock<Option<Arc<Network>>> {
    static SLOT: OnceLock<RwLock<Option<Arc<Network>>>> = OnceLock::new();
    SLOT.get_or_init(|| RwLock::new(Network::from_bytes(DEFAULT_NET).ok().map(Arc::new)))
}

/// The network new searches will use, or `None` if the embedded network failed to load and no other has been set.
pub fn current() -> Option<Arc<Network>> {
    slot().read().ok().and_then(|guard| guard.clone())
}

/// Replaces the active network. Searches that are already running keep the network they started with.
pub fn set_current(net: Option<Arc<Network>>) {
    if let Ok(mut guard) = slot().write() {
        *guard = net;
    }
}

/// Makes the network embedded in the binary the active one.
pub fn load_embedded() -> Result<(), LoadError> {
    let net = Network::from_bytes(DEFAULT_NET)?;
    set_current(Some(Arc::new(net)));
    Ok(())
}

/// Loads a network from `path` and makes it the active one.
///
/// # Errors
///
/// Returns a [`LoadError`] if the file cannot be read or is not a valid network.
/// On error the previously active network stays in place.
pub fn load_file(path: &str) -> Result<(), LoadError> {
    let bytes = std::fs::read(path).map_err(|e| LoadError::Io(format!("{path}: {e}")))?;
    let net = Network::from_bytes(&bytes)?;
    set_current(Some(Arc::new(net)));
    Ok(())
}

/// The vector implementation in use: `"avx512"`, `"avx2"` or `"scalar"`.
pub fn simd_description() -> &'static str {
    simd::description()
}

/// Vectorised kernels with a scalar fallback: add and subtract a weight column
/// to an accumulator, and the SCReLU dot product. The submodule that is used
/// depends on the compilation target; see the [module documentation](crate::nnue).
mod simd {
    use super::{HIDDEN, QA};

    pub mod scalar {
        use super::{HIDDEN, QA};

        pub fn add_assign(dst: &mut [i16; HIDDEN], col: &[i16]) {
            for (d, c) in dst.iter_mut().zip(col) {
                *d = d.wrapping_add(*c);
            }
        }

        #[cfg_attr(target_feature = "avx512bw", allow(dead_code))]
        pub fn sub_assign(dst: &mut [i16; HIDDEN], col: &[i16]) {
            for (d, c) in dst.iter_mut().zip(col) {
                *d = d.wrapping_sub(*c);
            }
        }

        pub fn screlu_dot(acc: &[i16; HIDDEN], weights: &[i16]) -> i32 {
            let mut sum = 0i32;
            for (a, w) in acc.iter().zip(weights) {
                let x = (*a).clamp(0, QA as i16) as i32;
                sum += x * *w as i32 * x;
            }
            sum
        }
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
    mod avx2 {
        use super::{HIDDEN, QA};
        use std::arch::x86_64::*;

        #[target_feature(enable = "avx2")]
        pub unsafe fn add_assign(dst: &mut [i16; HIDDEN], col: &[i16]) {
            unsafe {
                let mut i = 0;
                while i < HIDDEN {
                    let d = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
                    let c = _mm256_loadu_si256(col.as_ptr().add(i) as *const __m256i);
                    _mm256_storeu_si256(
                        dst.as_mut_ptr().add(i) as *mut __m256i,
                        _mm256_add_epi16(d, c),
                    );
                    i += 16;
                }
            }
        }

        #[target_feature(enable = "avx2")]
        pub unsafe fn sub_assign(dst: &mut [i16; HIDDEN], col: &[i16]) {
            unsafe {
                let mut i = 0;
                while i < HIDDEN {
                    let d = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
                    let c = _mm256_loadu_si256(col.as_ptr().add(i) as *const __m256i);
                    _mm256_storeu_si256(
                        dst.as_mut_ptr().add(i) as *mut __m256i,
                        _mm256_sub_epi16(d, c),
                    );
                    i += 16;
                }
            }
        }

        #[target_feature(enable = "avx2")]
        pub unsafe fn screlu_dot(acc: &[i16; HIDDEN], weights: &[i16]) -> i32 {
            unsafe {
                let zero = _mm256_setzero_si256();
                let limit = _mm256_set1_epi16(QA as i16);
                let mut sum = _mm256_setzero_si256();
                let mut i = 0;
                while i < HIDDEN {
                    let x = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
                    let x = _mm256_min_epi16(_mm256_max_epi16(x, zero), limit);
                    let w = _mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i);
                    let t = _mm256_mullo_epi16(x, w);
                    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(t, x));
                    i += 16;
                }
                let mut lanes = [0i32; 8];
                _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, sum);
                lanes.iter().sum()
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[allow(clippy::incompatible_msrv)]
    mod avx512 {
        use super::{HIDDEN, QA};
        use std::arch::x86_64::*;

        #[target_feature(enable = "avx512f,avx512bw")]
        pub unsafe fn add_assign(dst: &mut [i16; HIDDEN], col: &[i16]) {
            unsafe {
                let mut i = 0;
                while i < HIDDEN {
                    let d = _mm512_loadu_si512(dst.as_ptr().add(i) as *const _);
                    let c = _mm512_loadu_si512(col.as_ptr().add(i) as *const _);
                    _mm512_storeu_si512(dst.as_mut_ptr().add(i) as *mut _, _mm512_add_epi16(d, c));
                    i += 32;
                }
            }
        }

        #[target_feature(enable = "avx512f,avx512bw")]
        pub unsafe fn sub_assign(dst: &mut [i16; HIDDEN], col: &[i16]) {
            unsafe {
                let mut i = 0;
                while i < HIDDEN {
                    let d = _mm512_loadu_si512(dst.as_ptr().add(i) as *const _);
                    let c = _mm512_loadu_si512(col.as_ptr().add(i) as *const _);
                    _mm512_storeu_si512(dst.as_mut_ptr().add(i) as *mut _, _mm512_sub_epi16(d, c));
                    i += 32;
                }
            }
        }

        #[target_feature(enable = "avx512f,avx512bw")]
        pub unsafe fn screlu_dot(acc: &[i16; HIDDEN], weights: &[i16]) -> i32 {
            unsafe {
                let zero = _mm512_setzero_si512();
                let limit = _mm512_set1_epi16(QA as i16);
                let mut sum = _mm512_setzero_si512();
                let mut i = 0;
                while i < HIDDEN {
                    let x = _mm512_loadu_si512(acc.as_ptr().add(i) as *const _);
                    let x = _mm512_min_epi16(_mm512_max_epi16(x, zero), limit);
                    let w = _mm512_loadu_si512(weights.as_ptr().add(i) as *const _);
                    let t = _mm512_mullo_epi16(x, w);
                    sum = _mm512_add_epi32(sum, _mm512_madd_epi16(t, x));
                    i += 32;
                }
                _mm512_reduce_add_epi32(sum)
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    pub fn description() -> &'static str {
        "avx512"
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
    pub fn description() -> &'static str {
        if std::arch::is_x86_feature_detected!("avx2") {
            "avx2"
        } else {
            "scalar"
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn description() -> &'static str {
        "scalar"
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[inline(always)]
    pub fn add_assign(dst: &mut [i16; HIDDEN], col: &[i16]) {
        unsafe { avx512::add_assign(dst, col) }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[inline(always)]
    pub fn sub_assign(dst: &mut [i16; HIDDEN], col: &[i16]) {
        unsafe { avx512::sub_assign(dst, col) }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[inline(always)]
    pub fn screlu_dot(acc: &[i16; HIDDEN], weights: &[i16]) -> i32 {
        unsafe { avx512::screlu_dot(acc, weights) }
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
    #[inline(always)]
    pub fn add_assign(dst: &mut [i16; HIDDEN], col: &[i16]) {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { avx2::add_assign(dst, col) }
        } else {
            scalar::add_assign(dst, col)
        }
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
    #[inline(always)]
    pub fn sub_assign(dst: &mut [i16; HIDDEN], col: &[i16]) {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { avx2::sub_assign(dst, col) }
        } else {
            scalar::sub_assign(dst, col)
        }
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
    #[inline(always)]
    pub fn screlu_dot(acc: &[i16; HIDDEN], weights: &[i16]) -> i32 {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { avx2::screlu_dot(acc, weights) }
        } else {
            scalar::screlu_dot(acc, weights)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub use self::scalar::{add_assign, screlu_dot, sub_assign};
}
