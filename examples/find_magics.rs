//! Brute-force search for rook and bishop magic bitboard multipliers.
//!
//! Generates sparse 64-bit candidates per square, checks that the top byte of
//! `mask * magic` has enough bits set, and verifies that every blocker subset
//! hashes without a destructive collision. Prints ready-to-paste Rust arrays
//! for `src/magic_numbers.rs`.
//!
//! ```text
//! cargo run --release --example find_magics > magics.txt
//! ```

const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

fn on_board(file: i8, rank: i8) -> bool {
    (0..8).contains(&file) && (0..8).contains(&rank)
}

/// Reference ray walk used as the ground truth when checking candidate magics.
fn ray_attacks(sq: usize, occ: u64, dirs: &[(i8, i8)]) -> u64 {
    let mut out = 0u64;
    for &(dx, dy) in dirs {
        let mut f = (sq % 8) as i8 + dx;
        let mut r = (sq / 8) as i8 + dy;
        while on_board(f, r) {
            let bit = 1u64 << (r * 8 + f);
            out |= bit;
            if occ & bit != 0 {
                break;
            }
            f += dx;
            r += dy;
        }
    }
    out
}

/// Relevant blocker mask for `sq`, omitting board edge squares.
fn slider_mask(sq: usize, dirs: &[(i8, i8)]) -> u64 {
    let mut out = 0u64;
    for &(dx, dy) in dirs {
        let mut f = (sq % 8) as i8 + dx;
        let mut r = (sq / 8) as i8 + dy;
        while on_board(f, r) && on_board(f + dx, r + dy) {
            out |= 1u64 << (r * 8 + f);
            f += dx;
            r += dy;
        }
    }
    out
}

/// Software bit-deposit (`pdep`) to enumerate every subset of `mask`.
fn pdep(mut src: u64, mut mask: u64) -> u64 {
    let mut out = 0u64;
    while mask != 0 {
        let lowest = mask & mask.wrapping_neg();
        if src & 1 != 0 {
            out |= lowest;
        }
        src >>= 1;
        mask &= mask - 1;
    }
    out
}

/// XorShift64* RNG with a sparse-bit helper (ANDing three draws yields good magic candidates).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

/// Search until a valid magic multiplier is found for `sq`.
fn find_magic(sq: usize, dirs: &[(i8, i8)], rng: &mut Rng) -> u64 {
    let mask = slider_mask(sq, dirs);
    let bits = mask.count_ones();
    let size = 1usize << bits;
    let occupancies: Vec<u64> = (0..size).map(|i| pdep(i as u64, mask)).collect();
    let references: Vec<u64> = occupancies
        .iter()
        .map(|&o| ray_attacks(sq, o, dirs))
        .collect();
    let mut used = vec![(0u32, 0u64); size];
    let mut attempt = 0u32;

    loop {
        let magic = rng.sparse();
        if (mask.wrapping_mul(magic) >> 56).count_ones() < 6 {
            continue;
        }
        attempt += 1;
        let mut ok = true;
        for i in 0..size {
            let index = (occupancies[i].wrapping_mul(magic) >> (64 - bits)) as usize;
            let (stamp, value) = used[index];
            if stamp != attempt {
                used[index] = (attempt, references[i]);
            } else if value != references[i] {
                ok = false;
                break;
            }
        }
        if ok {
            return magic;
        }
    }
}

/// Format a 64-entry multiplier table as a Rust `pub const` array.
fn print_table(name: &str, magics: &[u64]) {
    println!("pub const {name}: [u64; 64] = [");
    for row in magics.chunks(2) {
        let cells: Vec<String> = row.iter().map(|m| format!("0x{m:016X}")).collect();
        println!("    {},", cells.join(", "));
    }
    println!("];");
}

fn main() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let rook: Vec<u64> = (0..64)
        .map(|sq| find_magic(sq, &ROOK_DIRS, &mut rng))
        .collect();
    let bishop: Vec<u64> = (0..64)
        .map(|sq| find_magic(sq, &BISHOP_DIRS, &mut rng))
        .collect();

    print_table("ROOK_MAGICS", &rook);
    println!();
    print_table("BISHOP_MAGICS", &bishop);
}
