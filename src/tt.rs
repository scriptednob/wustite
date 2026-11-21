//! Lockless shared transposition table.
//!
//! Caches search results by Zobrist hash so transpositions don't get re-searched
//! and the best move from shallower iterations gets tried first.
//!
//! # Lockless concurrent access
//!
//! Lazy SMP threads read and write the same table concurrently using
//! `Ordering::Relaxed` with no locks, relying on Robert Hyatt's XOR trick.
//! Each 16-byte slot holds two `AtomicU64` words:
//!
//! ```text
//! slot.data  = move | score | eval | depth | bound | age
//! slot.check = data ^ key
//! ```
//!
//! A probe loads both words and only accepts the slot when `check ^ data == key`.
//! If a concurrent write tears the two words, the XOR won't match and the probe
//! just treats it as a miss—never a corrupted score or illegal move.
//!
//! # Layout and replacement
//!
//! Four slots pack into a 64-byte, cache-line-aligned bucket. Buckets are
//! indexed via multiply-shift (`(key * buckets) >> 64`), which spreads keys
//! evenly without forcing power-of-two table sizes.
//!
//! On [`TranspositionTable::store`]:
//! 1. Fill any empty slot in the bucket first.
//! 2. If `key` is already in the bucket, overwrite it when the new entry is
//!    exact, within 3 plies of the old depth, or from a newer search (keeping
//!    the old best move if the new one is [`Move::NONE`]).
//! 3. Otherwise evict the slot minimizing `depth - 8 * relative_age`.
//!
//! Calling [`TranspositionTable::new_search`] bumps the 6-bit age between `go`
//! commands so old entries naturally get evicted without clearing the table.
//! Callers are responsible for converting mate scores to/from node-relative
//! distance before storing or using them (see [`crate::search`]).
//!
//! ```
//! use wustite::Move;
//! use wustite::tt::{Bound, TranspositionTable};
//!
//! let tt = TranspositionTable::new(1);
//! let key = 0x1234_5678_9abc_def0;
//! assert!(tt.probe(key).is_none());
//!
//! tt.store(key, Move::NONE, 42, 17, 6, Bound::Exact);
//! let entry = tt.probe(key).unwrap();
//! assert_eq!(entry.score, 42);
//! assert_eq!(entry.depth, 6);
//! assert_eq!(entry.bound, Bound::Exact);
//! ```

use core::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use crate::types::Move;

/// Meaning of a cached search score relative to the true value of the node.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Bound {
    /// Empty slot / no bound.
    None = 0,
    /// Fail-low: true score is at most `score`.
    Upper = 1,
    /// Fail-high: true score is at least `score`.
    Lower = 2,
    /// Exact score inside the `(alpha, beta)` window.
    Exact = 3,
}

impl Bound {
    /// Decode the 2-bit bound tag from a packed word.
    #[inline(always)]
    fn from_bits(bits: u64) -> Bound {
        match bits & 3 {
            1 => Bound::Upper,
            2 => Bound::Lower,
            3 => Bound::Exact,
            _ => Bound::None,
        }
    }
}

/// Unpacked transposition-table entry returned by [`TranspositionTable::probe`].
#[derive(Clone, Copy, Debug)]
pub struct TtEntry {
    /// Best or refutation move found, or [`Move::NONE`].
    pub mv: Move,
    /// Cached search score (mate scores are node-relative; callers re-base by ply).
    pub score: i32,
    /// Cached static eval so we don't recompute it on revisiting the node.
    pub eval: i32,
    /// Remaining search depth (quiescence entries use `0`).
    pub depth: i32,
    /// Whether `score` is exact, a lower bound, or an upper bound.
    pub bound: Bound,
}

/// Single lockless slot (`data` + `data ^ key` check word).
struct Slot {
    check: AtomicU64,
    data: AtomicU64,
}

/// Four slots packed into one 64-byte cache line.
#[repr(align(64))]
struct Bucket {
    slots: [Slot; 4],
}

impl Bucket {
    fn new() -> Bucket {
        Bucket {
            slots: [
                Slot {
                    check: AtomicU64::new(0),
                    data: AtomicU64::new(0),
                },
                Slot {
                    check: AtomicU64::new(0),
                    data: AtomicU64::new(0),
                },
                Slot {
                    check: AtomicU64::new(0),
                    data: AtomicU64::new(0),
                },
                Slot {
                    check: AtomicU64::new(0),
                    data: AtomicU64::new(0),
                },
            ],
        }
    }
}

/// Pack an entry into 64 bits: move (0..16), score (16..32), eval (32..48), depth (48..56), bound (56..58), age (58..64).
fn pack(mv: Move, score: i32, eval: i32, depth: i32, bound: Bound, age: u8) -> u64 {
    (mv.raw() as u64)
        | (((score as i16) as u16 as u64) << 16)
        | (((eval as i16) as u16 as u64) << 32)
        | ((depth.clamp(0, 255) as u64) << 48)
        | ((bound as u64) << 56)
        | (((age & 63) as u64) << 58)
}

/// Unpack a 64-bit data word into `(TtEntry, age)`.
fn unpack(data: u64) -> (TtEntry, u8) {
    let entry = TtEntry {
        mv: Move::from_raw(data as u16),
        score: ((data >> 16) as u16 as i16) as i32,
        eval: ((data >> 32) as u16 as i16) as i32,
        depth: ((data >> 48) & 0xFF) as i32,
        bound: Bound::from_bits(data >> 56),
    };
    (entry, ((data >> 58) & 63) as u8)
}

/// Fixed-size, lockless transposition table safe to share across threads via `&self`.
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use wustite::Move;
/// use wustite::tt::{Bound, TranspositionTable};
///
/// let tt = Arc::new(TranspositionTable::new(1));
/// let handles: Vec<_> = (0..4u64)
///     .map(|t| {
///         let tt = Arc::clone(&tt);
///         thread::spawn(move || {
///             for i in 0..10_000u64 {
///                 let key = (i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ t;
///                 tt.store(key, Move::NONE, 0, 0, 1, Bound::Exact);
///                 let _ = tt.probe(key);
///             }
///         })
///     })
///     .collect();
/// for handle in handles {
///     handle.join().unwrap();
/// }
/// ```
pub struct TranspositionTable {
    buckets: Vec<Bucket>,
    age: AtomicU8,
}

impl TranspositionTable {
    /// Allocate a zeroed table of roughly `megabytes` MiB (clamped to at least 1 MiB).
    pub fn new(megabytes: usize) -> TranspositionTable {
        let count = (megabytes.max(1) * 1024 * 1024 / core::mem::size_of::<Bucket>()).max(1);
        let mut buckets = Vec::with_capacity(count);
        buckets.resize_with(count, Bucket::new);
        TranspositionTable {
            buckets,
            age: AtomicU8::new(0),
        }
    }

    /// Zero every slot and reset the search age. Call only when no search is running.
    pub fn clear(&self) {
        for bucket in &self.buckets {
            for slot in &bucket.slots {
                slot.check.store(0, Ordering::Relaxed);
                slot.data.store(0, Ordering::Relaxed);
            }
        }
        self.age.store(0, Ordering::Relaxed);
    }

    /// Increment the 6-bit generation counter once per `go` so stale entries age out.
    pub fn new_search(&self) {
        let next = (self.age.load(Ordering::Relaxed) + 1) & 63;
        self.age.store(next, Ordering::Relaxed);
    }

    /// Map a 64-bit hash onto `0..buckets.len()` via multiply-shift.
    #[inline(always)]
    fn index(&self, key: u64) -> usize {
        ((key as u128 * self.buckets.len() as u128) >> 64) as usize
    }

    /// Issue an L1 prefetch hint for `key`'s bucket on x86-64 (no-op elsewhere).
    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        {
            let bucket = &self.buckets[self.index(key)];
            unsafe {
                core::arch::x86_64::_mm_prefetch(
                    bucket as *const Bucket as *const i8,
                    core::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = key;
    }

    /// Look up `key` in its bucket, returning `None` on a miss or a torn concurrent read.
    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<TtEntry> {
        let bucket = &self.buckets[self.index(key)];
        for slot in &bucket.slots {
            let data = slot.data.load(Ordering::Relaxed);
            let check = slot.check.load(Ordering::Relaxed);
            if data != 0 && check ^ data == key {
                return Some(unpack(data).0);
            }
        }
        None
    }

    /// Insert or update an entry for `key`, preserving any existing move if `mv` is [`Move::NONE`].
    ///
    /// ```
    /// use wustite::Move;
    /// use wustite::tt::{Bound, TranspositionTable};
    ///
    /// let tt = TranspositionTable::new(1);
    /// tt.store(7, Move::NONE, 10, 0, 3, Bound::Lower);
    /// assert_eq!(tt.probe(7).unwrap().bound, Bound::Lower);
    /// ```
    pub fn store(&self, key: u64, mv: Move, score: i32, eval: i32, depth: i32, bound: Bound) {
        let bucket = &self.buckets[self.index(key)];
        let age = self.age.load(Ordering::Relaxed);
        let mut victim = 0usize;
        let mut victim_value = i32::MAX;

        for (i, slot) in bucket.slots.iter().enumerate() {
            let data = slot.data.load(Ordering::Relaxed);
            let check = slot.check.load(Ordering::Relaxed);
            if data == 0 {
                victim = i;
                break;
            }
            let (old, old_age) = unpack(data);
            if check ^ data == key {
                let kept = if mv.is_none() { old.mv } else { mv };
                if bound == Bound::Exact || depth + 4 > old.depth || old_age != age {
                    let packed = pack(kept, score, eval, depth, bound, age);
                    slot.data.store(packed, Ordering::Relaxed);
                    slot.check.store(packed ^ key, Ordering::Relaxed);
                }
                return;
            }
            let relative_age = ((64 + age - old_age) & 63) as i32;
            let value = old.depth - 8 * relative_age;
            if value < victim_value {
                victim_value = value;
                victim = i;
            }
        }

        let packed = pack(mv, score, eval, depth, bound, age);
        let slot = &bucket.slots[victim];
        slot.data.store(packed, Ordering::Relaxed);
        slot.check.store(packed ^ key, Ordering::Relaxed);
    }

    /// Approximate fill rate in permille (`0..=1000`) for UCI `hashfull`, sampled over the first 250 buckets.
    pub fn hashfull(&self) -> usize {
        let age = self.age.load(Ordering::Relaxed);
        let sample = self.buckets.len().min(250);
        let mut used = 0usize;
        for bucket in &self.buckets[..sample] {
            for slot in &bucket.slots {
                let data = slot.data.load(Ordering::Relaxed);
                if data != 0 && unpack(data).1 == age {
                    used += 1;
                }
            }
        }
        used * 1000 / (sample * 4)
    }
}
