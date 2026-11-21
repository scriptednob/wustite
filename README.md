<div align="center">

# Wustite

[![License: GPL v3](https://shieldcn.dev/github/license/scriptednob/wustite.svg)](LICENSE)
[![CI](https://shieldcn.dev/github/ci/scriptednob/wustite.svg)](https://github.com/scriptednob/wustite/actions/workflows/ci.yml)
[![Release](https://shieldcn.dev/github/release/scriptednob/wustite.svg)](https://github.com/scriptednob/wustite/releases/latest)
[![badge](https://shieldcn.dev/crates/wustite.svg?variant=branded&logo=docsdotrs&brand=crates)](https://crates.io/crates/wustite)
![docs.rs](https://shieldcn.dev/badge/docs.rs-wustite-4f46e5.svg?logo=docsdotrs)
![Rust 1.85+](https://shieldcn.dev/badge/rust-1.85+.svg?logo=docsdotrs&brand=rust)

</div>

Wustite is a UCI chess engine written in Rust with zero external crates. Its NNUE network is generated and trained from scratch using the Rust tools in `examples/`, and self-play matches are handled by a couple of small Python scripts in `tools/`.

If you want to browse the library internals, the rustdoc pages are published on [docs.rs](https://docs.rs/wustite).

## Strength

Wustite isn't on CCRL or any other rating list yet. These numbers come from self-play SPRT matches against earlier versions using `tools/match.py`:

| Match | Time control | Games | Elo |
| --- | --- | --- | --- |
| NNUE vs. classical PeSTO | 100 ms/move | 89 | +144 ±66 |
| Current net vs. first net | 60 ms/move | 147 | +105 ±48 |
| v0.2.0 vs. classical PeSTO | 100 ms/move | 100 | +269 ±74 |
| 2 threads vs. 1 thread | 100 ms/move | 100 | +131 ±49 |

## What's inside

The board uses bitboards, copy-make, and an incremental Zobrist hash. Move generation only emits legal moves straight into a stack-allocated list, so nothing hits the heap during search. Slider attacks come from compile-time magic bitboards, or BMI2 `pext` when built with BMI2 enabled.

Search is fail-soft PVS with iterative deepening, aspiration windows, quiescence search, and Lazy SMP. It uses null-move pruning (with verification at higher depths), reverse futility, futility, late move pruning, late move reductions, SEE pruning, internal iterative reduction, check extensions, and singular extensions. Moves are ordered by TT move, SEE-split captures (MVV-LVA), two killers, a counter move, butterfly history, and 1-/2-ply continuation history.

Evaluation is a `(768 -> 128) x 2 -> 1` SCReLU NNUE updated incrementally as moves are made, with AVX-512BW and AVX2 paths on x86-64. PeSTO stays in the repo as a fallback (and for bootstrapping training data), alongside a small mop-up rule for bare-king endings where the net doesn't see enough examples.

The transposition table is lockless (Hyatt's XOR trick) and shared between threads in 64-byte cache-line buckets. The UCI front end supports `Hash`, `Threads`, `Move Overhead`, `Ponder`, `UseNNUE`, `EvalFile`, and `Clear Hash`.

## Getting started

### Prebuilt binaries

Binaries for Windows, macOS, and Linux are up on the [releases page](https://github.com/scriptednob/wustite/releases). Those are portable builds so they run anywhere, though building locally for your CPU will be faster.

Linux and macOS installer:

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/scriptednob/wustite/releases/latest/download/wustite-installer.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://github.com/scriptednob/wustite/releases/latest/download/wustite-installer.ps1 | iex"
```

Or with Cargo:

```sh
cargo install wustite
```

### Building from source

Requires Rust 1.85 or newer:

```sh
cargo build --release
```

To target your own CPU (enables `pext` if you have BMI2, and AVX-512BW on Rust 1.89+):

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Watch out on AMD Zen 1 and Zen 2: `pext` is microcoded there and slower than magic multiplication. Disable BMI2 on those chips:

```sh
RUSTFLAGS="-C target-cpu=native -C target-feature=-bmi2" cargo build --release
```

Without extra flags, x86-64 builds use magic bitboards and detect AVX2 at runtime. Other architectures (including ARM) fall back to scalar code for the network.

## Usage

Wustite has no built-in GUI. Load it into a UCI front end like [Cute Chess](https://github.com/cutechess/cutechess), [Arena](http://www.playwitharena.de/), [En Croissant](https://github.com/franciscoBSalgueiro/en-croissant), or [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot), or talk to it straight from a terminal:

```text
wustite
uci
position startpos moves e2e4 e7e5
go movetime 1000
```

### UCI options

| Option | Default | What it does |
| --- | --- | --- |
| `Hash` | `16` | Transposition table size in MiB (1 to 65536) |
| `Threads` | `1` | Lazy SMP worker threads (1 to 256) |
| `Move Overhead` | `30` | Milliseconds held back per move for GUI/pipe lag |
| `Ponder` | `false` | Let the GUI send `go ponder` |
| `UseNNUE` | `true` | Turn off to fall back to the classical PeSTO eval |
| `EvalFile` | `<internal>` | Path to a `.nnue` file, or `<internal>` for the embedded net |
| `Clear Hash` | | Empty the transposition table |

Internal search switches and margins can also be changed through `setoption` using integer values (`0` = off, anything else = on), for instance `setoption name NMP value 0`. They aren't advertised in `uci` since they're mainly for A/B testing changes. See `wustite::search::Params` and `wustite::uci` for the full list.

### CLI and debug commands

Run directly from the shell:

| Command | What it does |
| --- | --- |
| `wustite bench [depth]` | Fixed-depth benchmark across built-in positions (default depth 12) |
| `wustite perft <depth> [fen]` | Count leaf nodes from `startpos` or a FEN |
| `wustite divide <depth> [fen]` | Same as `perft`, broken down per root move |
| `wustite suite [max-depth]` | Check the built-in perft positions (default max depth 5) |

Inside a UCI session you can also run `d` (prints the board, FEN, and hash), `eval` (static eval), `bench [depth]` (default 10), and `go perft <depth>`.

## Training a network

The network takes 768 binary inputs per side (own/enemy, piece type, square), maps them to 128 hidden units with SCReLU, and produces one centipawn score. Bare-king endings against a rook or queen skip the net and use the mop-up eval instead, since self-play doesn't generate enough of them for the net to convert cleanly.

Both the self-play generator and trainer live in `examples/` and only need Rust:

```sh
cargo run --release --example datagen -- --out data.txt --threads 4 --positions 2000000 --nodes 8000
cargo run --release --example train -- --data data.txt --out net.nnue
```

`datagen` plays fixed-node self-play games after 8 or 9 random opening plies and writes `fen;score;wdl` lines. It defaults to PeSTO; pass `--net FILE` to label games with an existing network. `train` fits the weights with Adam, writes `net.nnue` plus a float checkpoint (`net.nnue.params` for `--resume`), and reports how closely the quantized network matches the float weights.

Compiling `train` with `-C target-cpu=native` speeds it up noticeably. Training on a million positions takes a few minutes; generating them is the slow part (roughly 400 positions/sec on two cores).

If you ever want to regenerate the magic multipliers in `src/magic_numbers.rs`, run `cargo run --release --example find_magics`.

## Python tools

The scripts in `tools/` require Python 3 and [python-chess](https://github.com/niklasf/python-chess).

`tools/make_openings.py` creates a balanced opening book by playing 6 or 8 random plies, deduplicating positions, and keeping ones that evaluate within ±100 cp at depth 7:

```sh
python3 tools/make_openings.py target/release/wustite tools/openings.txt 200
```

`tools/match.py` runs paired engine-vs-engine matches (each opening played from both colors) and stops early once SPRT accepts H0 or H1:

```sh
python3 tools/match.py --bin target/release/wustite --openings tools/openings.txt --movetime 60 \
    --a "evalfile=net.nnue" --b "evalfile=nets/wustite.nnue" --games 400
```

Pass `--bin-b <path>` to pit two different binaries against each other, or `--no-sprt` to play out all `--games`. Run `python3 tools/match.py --help` to see every flag.

## Tests and docs

```sh
cargo test --release
cargo doc --no-deps --open
```

The tests check standard and random perft positions (cross-checked against python-chess), board invariants across move trees, incremental vs. full-refresh NNUE updates, SIMD vs. scalar parity, forced mates, and the UCI loop. Doc examples run as tests too.

## Credits

- [Chess Programming Wiki](https://www.chessprogramming.org/) for almost everything behind the engine
- Ronald Friederich's [PeSTO](https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function) tables for the classical eval
- [python-chess](https://github.com/niklasf/python-chess) for cross-checking move generation and running matches
- [Stockfish](https://github.com/official-stockfish/Stockfish) and other open-source engines whose ideas are documented in public
- [shieldcn](https://shieldcn.dev) for the README badges
- [cargo-dist](https://github.com/axodotdev/cargo-dist) for release builds

## License

Wustite is released under the [GNU General Public License v3.0](LICENSE).
