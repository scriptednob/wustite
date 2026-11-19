<div align="center">

# Wustite

[![License: GPL v3](https://shieldcn.dev/github/license/scriptednob/wustite.svg)](LICENSE)
[![CI](https://shieldcn.dev/github/ci/scriptednob/wustite.svg)](https://github.com/scriptednob/wustite/actions/workflows/ci.yml)
[![Release](https://shieldcn.dev/github/release/scriptednob/wustite.svg)](https://github.com/scriptednob/wustite/releases/latest)
[![badge](https://shieldcn.dev/crates/wustite.svg?variant=branded&logo=docsdotrs&brand=crates)](https://crates.io/crates/wustite)
![docs.rs](https://shieldcn.dev/badge/docs.rs-wustite-4f46e5.svg?logo=docsdotrs)
![Rust 1.85+](https://shieldcn.dev/badge/rust-1.85+.svg?logo=docsdotrs&brand=rust)

</div>

Wustite is an open source UCI chess engine written in Rust with no dependencies, using an NNUE evaluation that is trained entirely with the tools in this repository.

It is built to be read as well as played against. Every module of the library explains what it does and why, with runnable examples, and the whole reference is on [docs.rs](https://docs.rs/wustite).

## Strength

Wustite is not on any rating list yet. The only results so far come from matches against its own earlier versions, played with `tools/match.py` and judged with an SPRT.

| Test                                     | Time control | Games | Elo difference |
| ---------------------------------------- | ------------ | ----- | -------------- |
| NNUE against the classical eval          | 100 ms/move  | 89    | +144 ±66       |
| Current network against the first        | 60 ms/move   | 147   | +105 ±48       |
| Version 0.2.0 against the classical eval | 100 ms/move  | 100   | +269 ±74       |
| 2 threads against 1 thread               | 100 ms/move  | 100   | +131 ±49       |

## Features

| Area       | What it uses                                                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Board      | Bitboards, copy-make, incremental Zobrist hash, fully legal move generation with no allocation                                       |
| Attacks    | Magic bitboards, or PEXT when built with BMI2, with every table built at compile time                                                |
| Search     | Principal variation search, iterative deepening, aspiration windows, quiescence search, Lazy SMP                                     |
| Pruning    | Null move, reverse futility, futility, late move pruning, late move reductions, SEE pruning, internal iterative reduction            |
| Extensions | Check and singular extensions                                                                                                        |
| Ordering   | Transposition table move, SEE-sorted captures, killers, counter move, butterfly and continuation history                             |
| Evaluation | NNUE with 768 inputs per side, 128 hidden units and SCReLU, updated incrementally, with AVX2 and AVX-512 code. PeSTO as the fallback |
| Table      | Lockless shared transposition table with one cache line per bucket                                                                   |
| Protocol   | UCI with pondering, `Hash`, `Threads` and `Move Overhead`                                                                            |

## Getting started

### Precompiled binaries

Binaries for Windows, macOS and Linux are on the [releases page](https://github.com/scriptednob/wustite/releases). They are generic builds that run on any CPU of the right architecture. For the best speed, build from source for your own machine.

There are also installers. On Linux and macOS:

```
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/scriptednob/wustite/releases/latest/download/wustite-installer.sh | sh
```

On Windows, in PowerShell:

```
powershell -ExecutionPolicy Bypass -c "irm https://github.com/scriptednob/wustite/releases/latest/download/wustite-installer.ps1 | iex"
```

If you have Rust installed, `cargo install wustite` also works.

### Building from source

You need a stable Rust toolchain, version 1.85 or newer.

```
cargo build --release
```

The executable is `target/release/wustite`. To build for your own CPU:

```
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This turns on PEXT attack lookups when the CPU has BMI2, and the AVX-512 network code when it has AVX-512 (that part needs Rust 1.89 or newer). PEXT is slow on AMD Zen 1 and Zen 2, so on those CPUs add `-C target-feature=-bmi2`.

Without any flags the engine uses magic bitboards and picks AVX2 at runtime for the network. On other architectures, ARM included, it uses plain scalar code, which works but is slower.

### Usage

Wustite is not a complete chess program. It needs a UCI interface such as [Cute Chess](https://github.com/cutechess/cutechess), [Arena](http://www.playwitharena.de/) or [En Croissant](https://github.com/franciscoBSalgueiro/en-croissant). It also works with [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot).

You can also talk to it directly from a terminal:

```
wustite
uci
position startpos moves e2e4 e7e5
go movetime 1000
```

### UCI options

| Name          | Default      | Description                                                   |
| ------------- | ------------ | ------------------------------------------------------------- |
| Hash          | 16           | Transposition table size in MB (1 to 65536)                   |
| Threads       | 1            | Number of search threads (1 to 256)                           |
| Move Overhead | 30           | Time in ms to keep in reserve for each move                   |
| Ponder        | false        | Allow the GUI to send `go ponder`                             |
| UseNNUE       | true         | Use the network. When off, the engine uses the classical eval |
| EvalFile      | `<internal>` | Path to a network file. `<internal>` loads the built-in one   |
| Clear Hash    |              | Empty the transposition table                                 |

Every search switch and margin can also be set with `setoption`, for example `setoption name NMP value 0`. They are listed in the [`search`](https://docs.rs/wustite/latest/wustite/search/struct.Params.html) and [`uci`](https://docs.rs/wustite/latest/wustite/uci/index.html) documentation.

### Custom commands

These work from the command line, for example `wustite perft 5`.

| Command                | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| `bench [depth]`        | Search a fixed set of positions and print nodes and speed |
| `perft <depth> [fen]`  | Count leaf nodes at a given depth                         |
| `divide <depth> [fen]` | Same as `perft`, with a count for each root move          |
| `suite [depth]`        | Run the built-in perft suite and check the results        |

Inside a UCI session these are available as well: `d` prints the board, `eval` prints the static evaluation, `bench [depth]` runs the benchmark and `go perft <depth>` counts nodes.

## Documentation

The API reference is generated from the source with rustdoc and published on [docs.rs/wustite](https://docs.rs/wustite). To read it offline:

```
cargo doc --no-deps --open
```

The crate page has a module map and a walk-through of how one move goes from the GUI to the board. The chess-specific modules each start with a long explanation of the technique they implement:

| Module                                                                    | Read it for                                                              |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [`board`](https://docs.rs/wustite/latest/wustite/board/index.html)        | Copy-make, FEN parsing, what `make_move` updates                         |
| [`movegen`](https://docs.rs/wustite/latest/wustite/movegen/index.html)    | How legality is built into generation: pins, evasions, en passant        |
| [`attacks`](https://docs.rs/wustite/latest/wustite/attacks/index.html)    | Magic bitboards and PEXT                                                 |
| [`search`](https://docs.rs/wustite/latest/wustite/search/index.html)      | Every pruning, reduction and extension, with the formulas                |
| [`tt`](https://docs.rs/wustite/latest/wustite/tt/index.html)              | The lockless table and its replacement policy                            |
| [`nnue`](https://docs.rs/wustite/latest/wustite/nnue/index.html)          | The network, incremental updates, quantisation and the file format       |
| [`uci`](https://docs.rs/wustite/latest/wustite/uci/index.html)            | Every command, option and the threading model                            |

## Training a network

The network has 768 inputs for each side (piece, colour and square), 128 hidden units, SCReLU and a single output. The hidden layer is updated incrementally as moves are made. Positions where one side has only a king and the other a rook or queen skip the network and use a simple mop-up evaluation, because the network has not seen enough of them.

The data generator and the trainer are in `examples/` and need nothing besides Rust.

```
cargo run --release --example datagen -- --out data.txt --threads 4 --positions 2000000 --nodes 8000
cargo run --release --example train -- --data data.txt --out net.nnue
```

`datagen` plays self-play games at a fixed node count and writes `fen;score;result` lines. Pass `--net FILE` to label positions with a network instead of the classical evaluation. `train` fits a network with Adam, writes a `.nnue` file and prints how far the quantized network is from the float one. Building the examples with `target-cpu=native` makes training much faster. Training on a million positions takes a few minutes, but generating them is slow: about 400 positions per second on two cores.

Before replacing the built-in network (`nets/wustite.nnue`), play the new one against it with `tools/match.py` and wait for the SPRT to finish:

```
python3 tools/match.py --bin target/release/wustite --openings tools/openings.txt --movetime 60 \
    --a "evalfile=net.nnue" --b "evalfile=nets/wustite.nnue" --games 400
```

`tools/match.py` needs Python with [python-chess](https://github.com/niklasf/python-chess) installed.

## Tests

```
cargo test --release
```

Move generation is checked with perft on the standard positions and on several hundred random positions compared against python-chess. The network code is checked by comparing every incremental update with a full recomputation, and SIMD results with the scalar code. The examples in the documentation are run as tests too. Search changes are only kept after an SPRT match against a build without them.

## Acknowledgements

- [Chess Programming Wiki](https://www.chessprogramming.org/), for most of what I know about writing an engine
- [PeSTO](https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function) by Ronald Friederich, whose tables are the classical evaluation
- [python-chess](https://github.com/niklasf/python-chess), used for cross-checking move generation and for running matches
- [Stockfish](https://github.com/official-stockfish/Stockfish) and the other open source engines whose techniques are written up in public
- [shieldcn](https://shieldcn.dev), for the badges
- [cargo-dist](https://github.com/axodotdev/cargo-dist), for the release builds

## License

Wustite is released under the [GNU General Public License v3.0](LICENSE).
