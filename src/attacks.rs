//! Compile-time attack tables and ray geometry.
//!
//! Every table in this module is built by `const fn` evaluation and lives in
//! read-only memory, so there's no runtime initialization step.
//!
//! * **Leapers** ([`knight_attacks`], [`king_attacks`], [`pawn_attacks`]):
//!   plain 64-entry tables.
//! * **Sliders** ([`rook_attacks`], [`bishop_attacks`], [`queen_attacks`]):
//!   compact fancy tables (102,400 rook entries and 5,248 bishop entries),
//!   populated at compile time via hyperbola quintessence. By default lookups
//!   use the magic multipliers in [`crate::magic_numbers`]; when compiled on
//!   x86-64 with `+bmi2`, they use hardware `pext` instead (check [`USE_PEXT`]).
//!   A bad magic multiplier panics at compile time rather than producing wrong
//!   moves. Note that `pext` is microcoded on AMD Zen 1/2, where magic
//!   bitboards are faster.
//! * **Geometry** ([`between`], [`line`](fn@line)): 64x64 tables for pins and
//!   check evasions.
//!
//! ```
//! use wustite::attacks::{knight_attacks, rook_attacks, between};
//! use wustite::{Bitboard, Square};
//!
//! assert_eq!(knight_attacks(Square::G1).count(), 3);
//!
//! let blocker = Square::A4.bb();
//! let attacks = rook_attacks(Square::A1, blocker);
//! assert!(attacks.contains(Square::A4));
//! assert!(!attacks.contains(Square::A5));
//!
//! assert_eq!(between(Square::A1, Square::A4), Square::A2.bb() | Square::A3.bb());
//! assert!(rook_attacks(Square::A1, Bitboard::EMPTY).contains(Square::H1));
//! ```

use crate::bitboard::Bitboard;
use crate::magic_numbers::{BISHOP_MAGICS, ROOK_MAGICS};
use crate::types::{Color, PieceType, Square};

/// `true` when compiled for x86-64 with BMI2 so slider lookups use `pext` instead of magic multiplication.
pub const USE_PEXT: bool = cfg!(all(target_arch = "x86_64", target_feature = "bmi2"));

/// Knight `(file, rank)` jump offsets.
const KNIGHT_DELTAS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
/// King `(file, rank)` step offsets.
const KING_DELTAS: [(i8, i8); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];
const WHITE_PAWN_DELTAS: [(i8, i8); 2] = [(-1, 1), (1, 1)];
const BLACK_PAWN_DELTAS: [(i8, i8); 2] = [(-1, -1), (1, -1)];
const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ALL_DIRS: [(i8, i8); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];

const ROOK_TABLE_SIZE: usize = 102_400;
const BISHOP_TABLE_SIZE: usize = 5_248;

/// Check whether `(file, rank)` lies inside the 8x8 board.
const fn on_board(file: i8, rank: i8) -> bool {
    file >= 0 && file < 8 && rank >= 0 && rank < 8
}

/// Compute fixed-offset attack bits for a single square.
const fn leaper_attacks(sq: usize, deltas: &[(i8, i8)]) -> u64 {
    let file = (sq % 8) as i8;
    let rank = (sq / 8) as i8;
    let mut out = 0u64;
    let mut i = 0;
    while i < deltas.len() {
        let f = file + deltas[i].0;
        let r = rank + deltas[i].1;
        if on_board(f, r) {
            out |= 1u64 << (r * 8 + f);
        }
        i += 1;
    }
    out
}

/// Slow ray-walk reference used to build geometry tables and verify slider lookups in tests.
const fn ray_attacks(sq: usize, occ: u64, dirs: &[(i8, i8)]) -> u64 {
    let mut out = 0u64;
    let mut d = 0;
    while d < dirs.len() {
        let mut f = (sq % 8) as i8 + dirs[d].0;
        let mut r = (sq / 8) as i8 + dirs[d].1;
        while on_board(f, r) {
            let bit = 1u64 << (r * 8 + f);
            out |= bit;
            if occ & bit != 0 {
                break;
            }
            f += dirs[d].0;
            r += dirs[d].1;
        }
        d += 1;
    }
    out
}

/// Relevant blocker mask for a slider on `sq` (excludes board edge squares that can't block further rays).
const fn slider_mask(sq: usize, dirs: &[(i8, i8)]) -> u64 {
    let mut out = 0u64;
    let mut d = 0;
    while d < dirs.len() {
        let (dx, dy) = dirs[d];
        let mut f = (sq % 8) as i8 + dx;
        let mut r = (sq / 8) as i8 + dy;
        while on_board(f, r) && on_board(f + dx, r + dy) {
            out |= 1u64 << (r * 8 + f);
            f += dx;
            r += dy;
        }
        d += 1;
    }
    out
}

/// Hyperbola quintessence along one line (`((o - 2s) ^ rev(rev(o) - 2*rev(s))) & line`), used during compile-time table fill.
const fn hyperbola(occ: u64, line_mask: u64, slider: u64) -> u64 {
    let forward = occ & line_mask;
    let reverse = forward.reverse_bits();
    let forward = forward.wrapping_sub(slider);
    let reverse = reverse.wrapping_sub(slider.reverse_bits());
    (forward ^ reverse.reverse_bits()) & line_mask
}

/// Pair of full empty-board lines through `sq` for a rook or bishop.
const fn slider_lines(sq: usize, rook: bool) -> (u64, u64) {
    if rook {
        (
            ray_attacks(sq, 0, &[(0, 1), (0, -1)]),
            ray_attacks(sq, 0, &[(1, 0), (-1, 0)]),
        )
    } else {
        (
            ray_attacks(sq, 0, &[(1, 1), (-1, -1)]),
            ray_attacks(sq, 0, &[(1, -1), (-1, 1)]),
        )
    }
}

/// Build a 64-square attack table for a leaper.
const fn build_leaper_table(deltas: &[(i8, i8)]) -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        table[sq] = leaper_attacks(sq, deltas);
        sq += 1;
    }
    table
}

/// Precomputed knight attacks by square.
static KNIGHT_TABLE: [u64; 64] = build_leaper_table(&KNIGHT_DELTAS);
/// Precomputed king attacks by square.
static KING_TABLE: [u64; 64] = build_leaper_table(&KING_DELTAS);
/// Precomputed pawn capture masks indexed `[color][square]`.
static PAWN_TABLE: [[u64; 64]; 2] = [
    build_leaper_table(&WHITE_PAWN_DELTAS),
    build_leaper_table(&BLACK_PAWN_DELTAS),
];

/// Per-square lookup metadata for magic/PEXT slider tables.
#[derive(Clone, Copy)]
#[cfg_attr(all(target_arch = "x86_64", target_feature = "bmi2"), allow(dead_code))]
struct MagicEntry {
    mask: u64,
    magic: u64,
    offset: u32,
    shift: u32,
}

/// Flat shared attack table plus 64 per-square [`MagicEntry`] headers.
struct SliderTable<const N: usize> {
    entries: [MagicEntry; 64],
    attacks: [u64; N],
}

impl<const N: usize> SliderTable<N> {
    const fn build(rook: bool, dirs: &[(i8, i8)], magics: &[u64; 64]) -> SliderTable<N> {
        let empty = MagicEntry {
            mask: 0,
            magic: 0,
            offset: 0,
            shift: 0,
        };
        let mut entries = [empty; 64];
        let mut attacks = [0u64; N];
        let mut offset = 0usize;
        let mut sq = 0;
        while sq < 64 {
            let mask = slider_mask(sq, dirs);
            let bits = mask.count_ones();
            let size = 1usize << bits;
            entries[sq] = MagicEntry {
                mask,
                magic: magics[sq],
                offset: offset as u32,
                shift: 64 - bits,
            };
            let (line_a, line_b) = slider_lines(sq, rook);
            let slider = 1u64 << sq;
            let magic = magics[sq];
            let shift = 64 - bits;
            let mut occ = 0u64;
            let mut i = 0;
            while i < size {
                let index = if USE_PEXT {
                    i
                } else {
                    (occ.wrapping_mul(magic) >> shift) as usize
                };
                let reference = hyperbola(occ, line_a, slider) | hyperbola(occ, line_b, slider);
                let slot = attacks[offset + index];
                assert!(slot == 0 || slot == reference, "magic number collision");
                attacks[offset + index] = reference;
                occ = occ.wrapping_sub(mask) & mask;
                i += 1;
            }
            offset += size;
            sq += 1;
        }
        assert!(offset == N, "slider table size mismatch");
        SliderTable { entries, attacks }
    }

    #[inline(always)]
    fn lookup(&self, sq: Square, occ: u64) -> u64 {
        let entry = &self.entries[sq.idx()];
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        let index = { unsafe { core::arch::x86_64::_pext_u64(occ, entry.mask) as usize } };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
        let index = ((occ & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;
        self.attacks[entry.offset as usize + index]
    }
}

/// Compile-time rook attack table.
#[allow(long_running_const_eval)]
static ROOK_SLIDERS: SliderTable<ROOK_TABLE_SIZE> =
    SliderTable::build(true, &ROOK_DIRS, &ROOK_MAGICS);
/// Compile-time bishop attack table.
#[allow(long_running_const_eval)]
static BISHOP_SLIDERS: SliderTable<BISHOP_TABLE_SIZE> =
    SliderTable::build(false, &BISHOP_DIRS, &BISHOP_MAGICS);

/// Precomputed 64x64 `between` and `line` tables.
struct Geometry {
    between: [[u64; 64]; 64],
    line: [[u64; 64]; 64],
}

impl Geometry {
    const fn build() -> Geometry {
        let mut g = Geometry {
            between: [[0; 64]; 64],
            line: [[0; 64]; 64],
        };
        let mut a = 0usize;
        while a < 64 {
            let mut d = 0;
            while d < 8 {
                let (dx, dy) = ALL_DIRS[d];
                let both = [(dx, dy), (-dx, -dy)];
                let full = ray_attacks(a, 0, &both) | (1u64 << a);
                let mut f = (a % 8) as i8 + dx;
                let mut r = (a / 8) as i8 + dy;
                let mut between = 0u64;
                while on_board(f, r) {
                    let b = (r * 8 + f) as usize;
                    g.between[a][b] = between;
                    g.line[a][b] = full;
                    between |= 1u64 << b;
                    f += dx;
                    r += dy;
                }
                d += 1;
            }
            a += 1;
        }
        g
    }
}

/// Compile-time geometry tables.
static GEOMETRY: Geometry = Geometry::build();

/// Squares attacked by a knight on `sq`.
#[inline(always)]
pub fn knight_attacks(sq: Square) -> Bitboard {
    Bitboard(KNIGHT_TABLE[sq.idx()])
}

/// Squares attacked by a king on `sq`.
#[inline(always)]
pub fn king_attacks(sq: Square) -> Bitboard {
    Bitboard(KING_TABLE[sq.idx()])
}

/// Diagonal capture targets of a `color` pawn on `sq` (excludes pushes).
///
/// Works in reverse too: enemy pawns that attack `sq` sit on
/// `pawn_attacks(!color, sq)`.
///
/// ```
/// use wustite::attacks::pawn_attacks;
/// use wustite::{Color, Square};
///
/// let white = pawn_attacks(Color::White, Square::E4);
/// assert_eq!(white, Square::D5.bb() | Square::F5.bb());
/// assert_eq!(pawn_attacks(Color::White, Square::A2).count(), 1);
/// ```
#[inline(always)]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    Bitboard(PAWN_TABLE[color.idx()][sq.idx()])
}

/// Squares a rook on `sq` attacks for occupancy `occ`.
///
/// Includes the first blocker in each direction (friendly or enemy), so callers
/// mask out their own pieces. Whether `sq` itself is set in `occ` doesn't matter.
#[inline(always)]
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    Bitboard(ROOK_SLIDERS.lookup(sq, occ.0))
}

/// Squares a bishop on `sq` attacks for occupancy `occ`. Same conventions as [`rook_attacks`].
#[inline(always)]
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    Bitboard(BISHOP_SLIDERS.lookup(sq, occ.0))
}

/// Squares a queen on `sq` attacks (`rook_attacks | bishop_attacks`).
#[inline(always)]
pub fn queen_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    rook_attacks(sq, occ) | bishop_attacks(sq, occ)
}

/// Dispatch attack lookup for any non-pawn piece type. Returns empty for pawns (use [`pawn_attacks`]).
#[inline(always)]
pub fn attacks_bb(piece_type: PieceType, sq: Square, occ: Bitboard) -> Bitboard {
    match piece_type {
        PieceType::Knight => knight_attacks(sq),
        PieceType::Bishop => bishop_attacks(sq, occ),
        PieceType::Rook => rook_attacks(sq, occ),
        PieceType::Queen => queen_attacks(sq, occ),
        PieceType::King => king_attacks(sq),
        PieceType::Pawn => Bitboard::EMPTY,
    }
}

/// Squares strictly between `a` and `b` if they share a rank, file, or diagonal; otherwise empty.
#[inline(always)]
pub fn between(a: Square, b: Square) -> Bitboard {
    Bitboard(GEOMETRY.between[a.idx()][b.idx()])
}

/// Full rank, file, or diagonal passing through `a` and `b` (inclusive), or empty if unaligned.
#[inline(always)]
pub fn line(a: Square, b: Square) -> Bitboard {
    Bitboard(GEOMETRY.line[a.idx()][b.idx()])
}

#[cfg(test)]
mod tests {
    use super::*;

    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn sliders_match_reference_rays() {
        let mut rng = XorShift(0x1234_5678_9ABC_DEF1);
        for _ in 0..20_000 {
            let occ = match rng.next() % 3 {
                0 => rng.next(),
                1 => rng.next() & rng.next(),
                _ => rng.next() & rng.next() & rng.next(),
            };
            for i in 0..64 {
                let sq = Square::from_idx(i);
                assert_eq!(
                    rook_attacks(sq, Bitboard(occ)).0,
                    ray_attacks(i, occ, &ROOK_DIRS)
                );
                assert_eq!(
                    bishop_attacks(sq, Bitboard(occ)).0,
                    ray_attacks(i, occ, &BISHOP_DIRS)
                );
            }
        }
    }

    #[test]
    fn leaper_counts() {
        assert_eq!(knight_attacks(Square::A1).count(), 2);
        assert_eq!(knight_attacks(Square::D4).count(), 8);
        assert_eq!(king_attacks(Square::A1).count(), 3);
        assert_eq!(king_attacks(Square::E4).count(), 8);
        assert_eq!(pawn_attacks(Color::White, Square::E2).count(), 2);
        assert_eq!(pawn_attacks(Color::White, Square::A2).count(), 1);
        assert_eq!(pawn_attacks(Color::Black, Square::H7).count(), 1);
    }

    #[test]
    fn between_and_line() {
        assert_eq!(
            between(Square::A1, Square::A4),
            Square::A2.bb() | Square::A3.bb()
        );
        assert_eq!(between(Square::A1, Square::H8).count(), 6);
        assert!(between(Square::A1, Square::B3).is_empty());
        assert!(between(Square::A1, Square::A2).is_empty());
        assert_eq!(line(Square::C3, Square::E5), line(Square::A1, Square::H8));
        assert!(line(Square::A1, Square::B3).is_empty());
    }
}
