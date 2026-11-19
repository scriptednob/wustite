//! The lockless shared transposition table.
//!
//! A transposition table caches search results keyed by position hash, so a
//! position reached by a different move order (a *transposition*) is not searched
//! again, and so the best move found at a shallower depth can be tried first at
//! the next iteration. It is the largest single contributor to search strength
//! after alpha-beta itself.
//!
//! # Thread safety without locks
//!
//! All search threads (Lazy SMP) share **one** table and read and write it
//! concurrently with no mutex. This is sound because every slot is two
//! `AtomicU64` words and an entry is validated on read:
//!
//! ```text
//! slot.data  = move | score | eval | depth | bound | age   (64 bits)
//! slot.check = data XOR key
//! ```
//!
//! A reader loads both words and accepts the entry only if
//! `check ^ data == key`. If another thread was halfway through a write and the
//! two words belong to different entries, the XOR will not reproduce the probing
//! key (with overwhelming probability) and the probe simply reports a miss. A torn
//! read costs one lost hit, never a wrong answer. All accesses use
//! `Ordering::Relaxed`; no ordering between slots is needed.
//!
//! This is the same "XOR trick" popularised by Robert Hyatt's lockless hashing.
//!
//! # Memory layout
//!
//! * A **slot** is 16 bytes: two atomic words.
//! * A **bucket** is four slots, 64 bytes, aligned to 64 so that it is exactly
//!   one cache line. A probe touches one line.
//! * The bucket for a key is chosen with a multiply-shift
//!   (`(key * bucket_count) >> 64`), which maps uniformly onto any bucket count
//!   without needing a power-of-two table size, so `Hash` can be any number of
//!   megabytes.
//! * [`TranspositionTable::prefetch`] issues a cache prefetch for a key's bucket.
//!   The search calls it right after creating a child position, before doing
//!   other work, to hide memory latency.
//!
//! # Entry contents
//!
//! Each entry holds the best move, the search score, the static evaluation, the
//! depth searched, a [`Bound`] and a 6-bit age. Scores and evaluations are
//! stored as `i16`; the engine's score range (`+-32000`) fits.
//!
//! # Replacement policy
//!
//! On [`TranspositionTable::store`]:
//!
//! 1. An empty slot in the bucket is used first.
//! 2. If the key is already present its slot is rewritten, but only when the new
//!    result is exact, no more than 3 plies shallower than the old one, or the
//!    old entry is from an earlier search. A stored best move is kept if the new result
//!    has none.
//! 3. Otherwise the victim is the slot with the lowest `depth - 8 * relative_age`,
//!    so shallow and stale entries are evicted first, and entries from the current
//!    search are protected.
//!
//! The *age* increments once per `go` ([`TranspositionTable::new_search`]), so
//! nothing has to be cleared between moves.
//!
//! # Mate scores
//!
//! The table does not know about the ply at which it was probed, so mate scores
//! are stored relative to the node and adjusted on the way in and out (see
//! [`crate::search`]). Without that, a "mate in 3" found deep in one line would be
//! reported as the wrong distance when reached by a transposition.
//!
//! # Examples
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

/// What a stored score means relative to the true value of the position.
///
/// Alpha-beta search rarely computes exact values. A node that fails high only
/// proves a lower bound; one that fails low proves an upper bound; only a node
/// whose value lands strictly inside the search window is exact.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Bound {
    /// No information. Also the value of an empty slot.
    None = 0,
    /// The true score is **at most** the stored score. The search failed low (no move improved alpha).
    Upper = 1,
    /// The true score is **at least** the stored score. The search failed high (a move reached beta).
    Lower = 2,
    /// The stored score is the true score. The value fell inside the alpha-beta window.
    Exact = 3,
}

impl Bound {
    /// Decodes the two bound bits of a packed entry.
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

/// A decoded transposition-table entry returned by [`TranspositionTable::probe`].
#[derive(Clone, Copy, Debug)]
pub struct TtEntry {
    /// The best or refutation move found, or [`Move::NONE`]. Searched first to improve move ordering.
    pub mv: Move,
    /// The search score. Mate scores are stored relative to the node and must be re-based by the caller.
    pub score: i32,
    /// The static evaluation of the position, cached so it need not be recomputed. This can be reused even when the search score is not usable.
    pub eval: i32,
    /// The remaining depth the position was searched to. Quiescence results are stored with depth 0.
    pub depth: i32,
    /// Whether [`TtEntry::score`] is exact or only a bound.
    pub bound: Bound,
}

/// One entry: the packed data word and the verification word `data ^ key`. See the [module documentation](crate::tt).
struct Slot {
    check: AtomicU64,
    data: AtomicU64,
}

/// Four slots in one 64-byte cache line.
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

/// Packs an entry into 64 bits:
///
/// | Bits | Field |
/// | --- | --- |
/// | 0-15 | move |
/// | 16-31 | score (`i16`) |
/// | 32-47 | static eval (`i16`) |
/// | 48-55 | depth (clamped to 0..=255) |
/// | 56-57 | bound |
/// | 58-63 | age (0..=63) |
fn pack(mv: Move, score: i32, eval: i32, depth: i32, bound: Bound, age: u8) -> u64 {
    (mv.raw() as u64)
        | (((score as i16) as u16 as u64) << 16)
        | (((eval as i16) as u16 as u64) << 32)
        | ((depth.clamp(0, 255) as u64) << 48)
        | ((bound as u64) << 56)
        | (((age & 63) as u64) << 58)
}

/// Decodes a packed entry, returning it together with its age.
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

/// A fixed-size, lockless, thread-shared transposition table.
///
/// All methods take `&self`: the table is designed to be shared by reference
/// between search threads (the engine wraps it in an `Arc` so the UCI thread can
/// also resize and clear it).
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
    /// Allocates a table of about `megabytes` MiB, rounded down to whole 64-byte
    /// buckets. At least one bucket is always allocated, so `new(0)` behaves like
    /// `new(1)`.
    ///
    /// The memory is zeroed, which marks every slot empty.
    pub fn new(megabytes: usize) -> TranspositionTable {
        let count = (megabytes.max(1) * 1024 * 1024 / core::mem::size_of::<Bucket>()).max(1);
        let mut buckets = Vec::with_capacity(count);
        buckets.resize_with(count, Bucket::new);
        TranspositionTable {
            buckets,
            age: AtomicU8::new(0),
        }
    }

    /// Empties the whole table and resets the age. Not safe to call meaningfully during a search; the UCI layer stops the search first.
    pub fn clear(&self) {
        for bucket in &self.buckets {
            for slot in &bucket.slots {
                slot.check.store(0, Ordering::Relaxed);
                slot.data.store(0, Ordering::Relaxed);
            }
        }
        self.age.store(0, Ordering::Relaxed);
    }

    /// Advances the age by one (modulo 64). Call once at the start of every search so that entries from earlier searches become eviction candidates instead of being cleared.
    pub fn new_search(&self) {
        let next = (self.age.load(Ordering::Relaxed) + 1) & 63;
        self.age.store(next, Ordering::Relaxed);
    }

    /// Maps a hash to a bucket index using multiply-shift, which is uniform for any bucket count.
    #[inline(always)]
    fn index(&self, key: u64) -> usize {
        ((key as u128 * self.buckets.len() as u128) >> 64) as usize
    }

    /// Hints the CPU to load the bucket for `key` into cache. A no-op on targets other than x86-64.
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

    /// Looks up `key`. Returns the entry if one with a valid verification word is
    /// found in the key's bucket, and `None` otherwise (including for torn reads
    /// caused by a concurrent write).
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

    /// Records a search result for `key`.
    ///
    /// `mv` may be [`Move::NONE`]; when overwriting an existing entry for the same
    /// key the old move is then preserved. See the [module documentation](crate::tt)
    /// for the replacement policy.
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

    /// Estimates how full the table is in permille (0 to 1000), the value UCI
    /// reports as `hashfull`. It samples the first 250 buckets and counts only slots
    /// written during the current search (matching age).
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
