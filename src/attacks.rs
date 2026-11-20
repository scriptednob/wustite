//! Attack sets for every piece, computed from lookup tables built at compile time.
//!
//! Nothing in this module is computed at run time and there is no
//! initialisation function: all tables are `static` values produced by `const fn`
//! evaluation, so they live in the binary's read-only data, and the first call
//! costs the same as any other.
//!
//! # Leapers
//!
//! Knights, kings and pawns attack a fixed pattern from each square regardless of
//! other pieces. Their sets are stored as one `u64` per square (and per colour for
//! pawns). See [`knight_attacks`], [`king_attacks`] and [`pawn_attacks`].
//!
//! # Sliders
//!
//! Rooks and bishops depend on which squares are occupied along their rays, so
//! the answer is a function of the origin square and the occupancy. The engine
//! uses a perfect hash of the relevant occupancy bits, in one of two flavours
//! chosen when the crate is compiled:
//!
//! * **Magic bitboards** (default). The *relevant* occupancy for a square is the
//!   set of ray squares excluding the board edge (an edge square cannot block
//!   anything beyond it). Multiplying `occupancy & mask` by a per-square magic
//!   constant and shifting right by `64 - bits` gives an index into that square's
//!   slice of a shared attack table. The magic constants live in
//!   [`crate::magic_numbers`] and were found with `examples/find_magics.rs`.
//! * **PEXT** (when built with `-C target-feature=+bmi2` on x86-64). The BMI2
//!   instruction `pext` extracts exactly the mask bits of the occupancy and packs
//!   them into the low bits of a register. That packed value is itself a perfect
//!   index, so no multiplication or magic constant is needed. [`USE_PEXT`]
//!   reports which variant was compiled in. Note that `pext` is microcoded and
//!   slow on AMD Zen 1 and Zen 2, where the magic variant is faster.
//!
//! Table sizes are 102,400 entries for rooks and 5,248 for bishops (about 819 KB
//! and 42 KB of `u64`), the standard sizes for fully compact "fancy" magic
//! tables.
//!
//! The attack table contents are generated with *hyperbola quintessence*, a
//! subtraction-based trick that computes a line's attacks from an occupancy
//! without looping over squares, and `const` evaluation asserts that no two
//! occupancies that need different answers share an index. A bad magic constant
//! fails the build instead of silently producing wrong moves.
//!
//! # Geometry tables
//!
//! [`between`] and [`line`](fn@line) are two 64 x 64 tables used for pins and check
//! evasions: the squares strictly between two aligned squares, and the whole
//! board line through them.
//!
//! # Examples
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

/// Whether slider attacks use the BMI2 `pext` instruction instead of magic multiplication.
///
/// `true` only when the crate was compiled for x86-64 with the `bmi2` target
/// feature, for example `RUSTFLAGS="-C target-cpu=native"` on a CPU that has it.
pub const USE_PEXT: bool = cfg!(all(target_arch = "x86_64", target_feature = "bmi2"));

/// Knight move offsets as `(file, rank)` deltas.
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
/// King move offsets as `(file, rank)` deltas.
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

/// Whether a file and rank pair is inside the 8x8 board.
const fn on_board(file: i8, rank: i8) -> bool {
    file >= 0 && file < 8 && rank >= 0 && rank < 8
}

/// Attack set of a piece that jumps by fixed offsets, for one origin square.
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

/// Reference slider attacks for one square: walks each direction until it leaves
/// the board or hits an occupied square, which is included in the result.
///
/// This slow, obviously correct version is the oracle the unit tests compare the
/// fast lookup against.
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

/// The relevant occupancy mask for a slider on one square: every ray square
/// except the last one in each direction, because the final square on a ray can
/// never be a blocker that changes the result.
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

/// Hyperbola quintessence for a single line.
///
/// With `o` the occupancy on the line, `s` the slider bit and `r(x)` bit
/// reversal, the attacks along that line are `((o - 2s) ^ r(r(o) - 2r(s))) & line`.
/// The subtraction propagates a borrow from the slider up to the first blocker,
/// flipping exactly the squares in between; the reversed copy does the same in
/// the opposite direction. Used only to fill the tables at compile time.
const fn hyperbola(occ: u64, line_mask: u64, slider: u64) -> u64 {
    let forward = occ & line_mask;
    let reverse = forward.reverse_bits();
    let forward = forward.wrapping_sub(slider);
    let reverse = reverse.wrapping_sub(slider.reverse_bits());
    (forward ^ reverse.reverse_bits()) & line_mask
}

/// The two lines a rook (orthogonals) or bishop (diagonals) travels along from one square, used by [`hyperbola`].
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

/// Builds a 64-entry attack table for a leaper from its offsets.
const fn build_leaper_table(deltas: &[(i8, i8)]) -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        table[sq] = leaper_attacks(sq, deltas);
        sq += 1;
    }
    table
}

/// Knight attacks indexed by origin square.
static KNIGHT_TABLE: [u64; 64] = build_leaper_table(&KNIGHT_DELTAS);
/// King attacks indexed by origin square.
static KING_TABLE: [u64; 64] = build_leaper_table(&KING_DELTAS);
/// Pawn capture targets indexed by colour, then origin square.
static PAWN_TABLE: [[u64; 64]; 2] = [
    build_leaper_table(&WHITE_PAWN_DELTAS),
    build_leaper_table(&BLACK_PAWN_DELTAS),
];

/// Per-square slider lookup parameters: the relevant-occupancy mask, the magic
/// multiplier, the offset of this square's slice in the shared attack table, and
/// the right-shift that turns the product into an index.
#[derive(Clone, Copy)]
#[cfg_attr(all(target_arch = "x86_64", target_feature = "bmi2"), allow(dead_code))]
struct MagicEntry {
    mask: u64,
    magic: u64,
    offset: u32,
    shift: u32,
}

/// One complete slider lookup structure (rooks or bishops): 64 [`MagicEntry`]
/// records and a flat attack array of `N` entries shared by all squares.
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

/// The rook attack table, built at compile time.
#[allow(long_running_const_eval)]
static ROOK_SLIDERS: SliderTable<ROOK_TABLE_SIZE> =
    SliderTable::build(true, &ROOK_DIRS, &ROOK_MAGICS);
/// The bishop attack table, built at compile time.
#[allow(long_running_const_eval)]
static BISHOP_SLIDERS: SliderTable<BISHOP_TABLE_SIZE> =
    SliderTable::build(false, &BISHOP_DIRS, &BISHOP_MAGICS);

/// The `between` and `line` tables, each 64 x 64 `u64` values.
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

/// The geometry tables, built at compile time.
static GEOMETRY: Geometry = Geometry::build();

/// Squares a knight on `sq` attacks.
#[inline(always)]
pub fn knight_attacks(sq: Square) -> Bitboard {
    Bitboard(KNIGHT_TABLE[sq.idx()])
}

/// Squares a king on `sq` attacks.
#[inline(always)]
pub fn king_attacks(sq: Square) -> Bitboard {
    Bitboard(KING_TABLE[sq.idx()])
}

/// Squares a pawn of `color` on `sq` attacks (its two diagonal capture targets,
/// fewer on the edge files). The result does not include push squares.
///
/// The same table answers the reverse question: the squares from which an enemy
/// pawn would attack `sq` are `pawn_attacks(!color, sq)`. Position code uses that
/// to find pawn attackers without a loop.
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

/// Squares a rook on `sq` attacks given the occupancy `occ`.
///
/// Each ray extends up to and including the first occupied square, so the result
/// includes blockers of either colour; callers mask out their own pieces.
/// `occ` may or may not include the rook's own square, since it is not part of
/// the relevant mask.
#[inline(always)]
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    Bitboard(ROOK_SLIDERS.lookup(sq, occ.0))
}

/// Squares a bishop on `sq` attacks given the occupancy `occ`. Same conventions as [`rook_attacks`].
#[inline(always)]
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    Bitboard(BISHOP_SLIDERS.lookup(sq, occ.0))
}

/// Squares a queen on `sq` attacks: the union of [`rook_attacks`] and [`bishop_attacks`].
#[inline(always)]
pub fn queen_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    rook_attacks(sq, occ) | bishop_attacks(sq, occ)
}

/// Attacks of a non-pawn piece type from `sq`, dispatching to the specific lookup.
///
/// Pawns return the empty set because their attacks depend on colour; use
/// [`pawn_attacks`] for them. This function is what move generation calls for
/// knights, bishops, rooks and queens in a single loop.
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

/// The squares strictly between `a` and `b` when they share a rank, file or
/// diagonal, and the empty set otherwise (including when they are adjacent).
///
/// Used to find which pieces block a check or pin a piece, and which squares a
/// checked king's side may interpose on.
#[inline(always)]
pub fn between(a: Square, b: Square) -> Bitboard {
    Bitboard(GEOMETRY.between[a.idx()][b.idx()])
}

/// The entire board line (rank, file or diagonal) through `a` and `b`, including
/// both squares, or the empty set when they are not aligned.
///
/// A piece pinned to its king may move only along this line, so move generation
/// intersects a pinned piece's destinations with `line(king, piece)`.
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
