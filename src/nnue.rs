//! Incrementally updated NNUE evaluation (`(768 -> 128) x 2 -> 1`, SCReLU).
//!
//! Uses a compact two-perspective architecture trained with `examples/datagen.rs`
//! and `examples/train.rs`.
//!
//! # Architecture and incremental updates
//!
//! Each perspective has 768 binary inputs: `(enemy * 6 + piece_type) * 64 + sq`,
//! with squares flipped vertically (`sq ^ 56`) for black so both views share
//! one feature-transformer matrix. At inference time the side-to-move's
//! accumulator multiplies the first 128 output weights ("us") and the opponent's
//! multiplies the second 128 ("them"), using squared clipped ReLU
//! (`clamp(x, 0, QA)^2`).
//!
//! Because the first layer is just a sum of active feature columns plus bias,
//! [`update`] doesn't recompute it from scratch on every move—it copies the
//! parent [`Accumulator`] and adds/subtracts the 2 to 5 features that changed.
//!
//! # Quantization and SIMD
//!
//! All inference runs in integer arithmetic (`QA = 255`, `QB = 64`,
//! `SCALE = 400`). Output weights are capped at `+-127` ([`OUTPUT_LIMIT`]) so
//! `255 * 127 = 32385` fits in an `i16` without overflowing 16-bit SIMD lanes;
//! [`Network::from_bytes`] rejects files that exceed this bound.
//!
//! Kernels dispatch to AVX-512BW (compile-time on Rust 1.89+), AVX2 (runtime
//! check on x86-64), or portable scalar code. Check [`simd_description`] to see
//! which path is active.
//!
//! # File format (`WNNU` v1, little-endian, 197,396 bytes)
//!
//! | Offset | Size | Content |
//! | --- | --- | --- |
//! | 0 | 4 | magic `"WNNU"` ([`MAGIC`]) |
//! | 4 | 4 | version `u32` ([`VERSION`]) |
//! | 8 | 4 | inputs `u32` (768) |
//! | 12 | 4 | hidden size `u32` (128) |
//! | 16 | `2 * 768 * 128` | feature-transformer weights (`i16`, column-major by feature) |
//! | next | `2 * 128` | feature-transformer bias (`i16`) |
//! | next | `2 * 256` | output weights (`i16`, "us" then "them") |
//! | last | 4 | output bias (`i32`) |
//!
//! The default net (`nets/wustite.nnue`) is compiled into the binary and loaded
//! lazily into a process-wide slot replaceable with [`load_file`].
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
//! Incremental updates match a full refresh bit-for-bit:
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

/// Binary input features per perspective (`2 * 6 * 64`).
pub const INPUTS: usize = 768;
/// Hidden units per perspective.
pub const HIDDEN: usize = 128;
/// Quantization scale for feature-transformer activations (`0..=QA`).
pub const QA: i32 = 255;
/// Quantization scale for output layer weights.
pub const QB: i32 = 64;
/// Centipawn scale factor applied at the output.
pub const SCALE: i32 = 400;
/// Max allowed `|output_weight|` so 16-bit SIMD `clamp(a, 0, 255) * w` never overflows.
pub const OUTPUT_LIMIT: i16 = 127;
/// Four-byte file header magic (`WNNU`).
pub const MAGIC: [u8; 4] = *b"WNNU";
/// Supported `WNNU` format version.
pub const VERSION: u32 = 1;
/// Header size in bytes (magic, version, inputs, hidden).
pub const HEADER_BYTES: usize = 16;
/// Exact byte length of a valid v1 network file (197,396 bytes).
pub const FILE_BYTES: usize = HEADER_BYTES + 2 * (INPUTS * HIDDEN + HIDDEN + 2 * HIDDEN) + 4;

/// Built-in network embedded from `nets/wustite.nnue`.
static DEFAULT_NET: &[u8] = include_bytes!("../nets/wustite.nnue");

/// Error returned when loading or validating a `.nnue` network fails.
///
/// ```
/// use wustite::nnue::{LoadError, Network};
///
/// assert!(matches!(Network::from_bytes(&[0u8; 4]), Err(LoadError::Size { .. })));
/// assert!(matches!(Network::from_bytes(&[0u8; 16]), Err(LoadError::Magic)));
/// ```
#[derive(Debug)]
pub enum LoadError {
    /// Input length didn't match [`FILE_BYTES`].
    Size {
        /// Expected byte length ([`FILE_BYTES`]).
        expected: usize,
        /// Actual byte length read.
        found: usize,
    },
    /// Missing `WNNU` magic bytes.
    Magic,
    /// Unsupported format version.
    Version(u32),
    /// Network dimensions don't match `768x128`.
    Shape {
        /// Input count from the file header.
        inputs: u32,
        /// Hidden layer width from the file header.
        hidden: u32,
    },
    /// An output weight exceeded `+-OUTPUT_LIMIT`.
    OutputRange,
    /// Filesystem read error (message includes the path).
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

/// Immutable, quantized NNUE weights shared between search threads via `Arc`.
pub struct Network {
    ft_weights: Box<[i16]>,
    ft_bias: [i16; HIDDEN],
    out_weights: [i16; 2 * HIDDEN],
    out_bias: i32,
}

/// 64-byte aligned hidden-layer accumulators for both perspectives (`0` = white, `1` = black).
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    /// Hidden unit sums indexed as `v[perspective.idx()][unit]`.
    pub v: [[i16; HIDDEN]; 2],
}

impl Default for Accumulator {
    fn default() -> Accumulator {
        Accumulator {
            v: [[0; HIDDEN]; 2],
        }
    }
}

/// Read a little-endian `i16` at `offset`.
fn read_i16(bytes: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

/// Read a little-endian `u32` at `offset`.
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Pack raw quantized weights into the `WNNU` v1 binary format (used by `examples/train.rs`).
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
    /// Parse and validate a `WNNU` v1 byte buffer into a [`Network`].
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

    /// Serialize this network back into `WNNU` v1 bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        serialize(
            &self.ft_weights,
            &self.ft_bias,
            &self.out_weights,
            self.out_bias,
        )
    }

    /// Slice of 128 feature-transformer weights for one input feature.
    #[inline(always)]
    fn column(&self, feature: usize) -> &[i16] {
        &self.ft_weights[feature * HIDDEN..(feature + 1) * HIDDEN]
    }
}

/// Map `(perspective, piece, sq)` to an input index in `0..768`.
///
/// Mirrors ranks (`sq ^ 56`) and swaps own/enemy when `perspective == Black` so
/// one set of weights covers both sides.
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

/// Recompute both perspectives of `acc` from scratch for `pos` (used at the search root).
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

/// Scalar-only [`refresh`], used as a reference in tests and debug assertions.
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

/// Incrementally derive `dst` from parent accumulator `src` for legal move `mv` played on `pos`.
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

/// Rescale the raw SCReLU dot product and add `out_bias` to produce centipawns.
#[inline(always)]
fn finish(net: &Network, sum: i64) -> i32 {
    let hidden = sum / QA as i64 + net.out_bias as i64;
    (hidden * SCALE as i64 / (QA * QB) as i64) as i32
}

/// Score a position in centipawns from `stm`'s perspective using its up-to-date accumulator.
pub fn evaluate(net: &Network, acc: &Accumulator, stm: Color) -> i32 {
    let us = &acc.v[stm.idx()];
    let them = &acc.v[stm.flip().idx()];
    let sum = simd::screlu_dot(us, &net.out_weights[..HIDDEN]) as i64
        + simd::screlu_dot(them, &net.out_weights[HIDDEN..]) as i64;
    finish(net, sum)
}

/// Scalar-only [`evaluate`], used to verify bit-exact SIMD parity in tests.
pub fn scalar_evaluate(net: &Network, acc: &Accumulator, stm: Color) -> i32 {
    let us = &acc.v[stm.idx()];
    let them = &acc.v[stm.flip().idx()];
    let sum = simd::scalar::screlu_dot(us, &net.out_weights[..HIDDEN]) as i64
        + simd::scalar::screlu_dot(them, &net.out_weights[HIDDEN..]) as i64;
    finish(net, sum)
}

/// Lazily initialized global slot holding the active network.
fn slot() -> &'static RwLock<Option<Arc<Network>>> {
    static SLOT: OnceLock<RwLock<Option<Arc<Network>>>> = OnceLock::new();
    SLOT.get_or_init(|| RwLock::new(Network::from_bytes(DEFAULT_NET).ok().map(Arc::new)))
}

/// Grab a handle to the currently active network, if one is loaded.
pub fn current() -> Option<Arc<Network>> {
    slot().read().ok().and_then(|guard| guard.clone())
}

/// Swap the active network for future searches (in-flight searches keep their `Arc`).
pub fn set_current(net: Option<Arc<Network>>) {
    if let Ok(mut guard) = slot().write() {
        *guard = net;
    }
}

/// Restore the built-in network embedded in the binary.
pub fn load_embedded() -> Result<(), LoadError> {
    let net = Network::from_bytes(DEFAULT_NET)?;
    set_current(Some(Arc::new(net)));
    Ok(())
}

/// Load a `.nnue` file from disk and make it active (leaves the old net untouched on error).
pub fn load_file(path: &str) -> Result<(), LoadError> {
    let bytes = std::fs::read(path).map_err(|e| LoadError::Io(format!("{path}: {e}")))?;
    let net = Network::from_bytes(&bytes)?;
    set_current(Some(Arc::new(net)));
    Ok(())
}

/// Active kernel name: `"avx512"`, `"avx2"`, or `"scalar"`.
pub fn simd_description() -> &'static str {
    simd::description()
}

/// Vectorized `add_assign`, `sub_assign`, and `screlu_dot` kernels with scalar fallback.
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
