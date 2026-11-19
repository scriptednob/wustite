//! The classical hand-written evaluation, and the bare-king "mop-up" rule.
//!
//! This module is the engine's fallback evaluation. The default evaluator is the
//! NNUE network in [`crate::nnue`]; this code is used when the network is turned
//! off (`UseNNUE=false`), when no network could be loaded, and as the labelling
//! function for the first round of training data.
//!
//! All scores are in centipawns from the point of view of the **side to move**:
//! positive means the player to move is better. That "negamax convention" lets
//! the search flip the sign at each ply without tracking colour.
//!
//! # PeSTO
//!
//! [`evaluate`] is Ronald Friederich's PeSTO ("Piece-Square Tables Only")
//! evaluation. Every piece type has a material value and a 64-entry
//! piece-square table, with **separate middlegame and endgame versions**. The two
//! scores are blended by a *game phase* computed from the material left on the
//! board (knight 1, bishop 1, rook 2, queen 4, total 24 in the starting
//! position):
//!
//! ```text
//! score = (mg * phase + eg * (24 - phase)) / 24
//! ```
//!
//! So a position with all pieces on is scored by the middlegame tables, a bare
//! pawn ending by the endgame tables, and anything in between by interpolation.
//! This "tapered" evaluation avoids the abrupt jumps of a hard middlegame/endgame
//! switch. A small tempo bonus is added for the side to move.
//!
//! Tables are written from white's point of view with rank 8 first, and a
//! compile-time `const fn` expands them into per-piece, per-square arrays
//! (value plus piece-square bonus) for both colours, so evaluating a piece is one
//! array read.
//!
//! # Mop-up
//!
//! [`bare_king_eval`] handles positions where one side has only a king and the
//! other has at least a rook or queen. The neural network has essentially never
//! seen such positions and its output there is noisy enough that pruning could
//! hide a short mate. The mop-up rule is a smooth, monotonic gradient that
//! steers the search toward the standard mating technique: push the bare king
//! to the edge and bring the winning king close.
//!
//! # Examples
//!
//! ```
//! use wustite::Position;
//! use wustite::eval::{bare_king_eval, evaluate};
//!
//! let start = Position::startpos();
//! assert!(evaluate(&start).abs() < 50);
//!
//! let up_a_queen: Position = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1".parse().unwrap();
//! assert!(evaluate(&up_a_queen) > 800);
//! assert!(bare_king_eval(&up_a_queen).is_some());
//! assert!(bare_king_eval(&start).is_none());
//! ```

use crate::board::Position;
use crate::types::{Color, Piece, PieceType};

/// How much each piece type (pawn through king) contributes to the game phase.
const PHASE_WEIGHT: [i32; 6] = [0, 1, 1, 2, 4, 0];
/// The phase of the starting position: 2 x (2 knights + 2 bishops) + 2 x 2 rooks x 2 + 2 queens x 4 = 24.
const TOTAL_PHASE: i32 = 24;
/// Bonus, in centipawns, for having the move.
const TEMPO: i32 = 10;

/// Middlegame material values in centipawns (pawn, knight, bishop, rook, queen, king).
const MG_VALUE: [i32; 6] = [82, 337, 365, 477, 1025, 0];
/// Endgame material values in centipawns. Also used by [`bare_king_eval`] as the material term.
const EG_VALUE: [i32; 6] = [94, 281, 297, 512, 936, 0];

/// Middlegame piece-square tables, indexed `[piece type][square]`, written with rank 8 first.
#[rustfmt::skip]
const MG_PST: [[i32; 64]; 6] = [
    [
          0,   0,   0,   0,   0,   0,  0,   0,
         98, 134,  61,  95,  68, 126, 34, -11,
         -6,   7,  26,  31,  65,  56, 25, -20,
        -14,  13,   6,  21,  23,  12, 17, -23,
        -27,  -2,  -5,  12,  17,   6, 10, -25,
        -26,  -4,  -4, -10,   3,   3, 33, -12,
        -35,  -1, -20, -23, -15,  24, 38, -22,
          0,   0,   0,   0,   0,   0,  0,   0,
    ],
    [
        -167, -89, -34, -49,  61, -97, -15, -107,
         -73, -41,  72,  36,  23,  62,   7,  -17,
         -47,  60,  37,  65,  84, 129,  73,   44,
          -9,  17,  19,  53,  37,  69,  18,   22,
         -13,   4,  16,  13,  28,  19,  21,   -8,
         -23,  -9,  12,  10,  19,  17,  25,  -16,
         -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    ],
    [
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
         -4,   5,  19,  50,  37,  37,   7,  -2,
         -6,  13,  13,  26,  34,  12,  10,   4,
          0,  15,  15,  15,  14,  27,  18,  10,
          4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    ],
    [
         32,  42,  32,  51, 63,  9,  31,  43,
         27,  32,  58,  62, 80, 67,  26,  44,
         -5,  19,  26,  36, 17, 45,  61,  16,
        -24, -11,   7,  26, 24, 35,  -8, -20,
        -36, -26, -12,  -1,  9, -7,   6, -23,
        -45, -25, -16, -17,  3,  0,  -5, -33,
        -44, -16, -20,  -9, -1, 11,  -6, -71,
        -19, -13,   1,  17, 16,  7, -37, -26,
    ],
    [
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
         -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
         -1, -18,  -9,  10, -15, -25, -31, -50,
    ],
    [
        -65,  23,  16, -15, -56, -34,   2,  13,
         29,  -1, -20,  -7,  -8,  -4, -38, -29,
         -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
          1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    ],
];

/// Endgame piece-square tables, same layout as [`MG_PST`].
#[rustfmt::skip]
const EG_PST: [[i32; 64]; 6] = [
    [
          0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
         94, 100,  85,  67,  56,  53,  82,  84,
         32,  24,  13,   5,  -2,   4,  17,  17,
         13,   9,  -3,  -7,  -7,  -8,   3,  -1,
          4,   7,  -6,   1,   0,  -5,  -1,  -8,
         13,   8,   8,  10,  13,   0,   2,  -7,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    [
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ],
    [
        -14, -21, -11,  -8, -7,  -9, -17, -24,
         -8,  -4,   7, -12, -3, -13,  -4, -14,
          2,  -8,   0,  -1, -2,   6,   0,   4,
         -3,   9,  12,   9, 14,  10,   3,   2,
         -6,   3,  13,  19,  7,  10,  -3,  -9,
        -12,  -3,   8,  10, 13,   3,  -7, -15,
        -14, -18,  -7,  -1,  4,  -9, -15, -27,
        -23,  -9, -23,  -5, -9, -16,  -5, -17,
    ],
    [
        13, 10, 18, 15, 12,  12,   8,   5,
        11, 13, 13, 11, -3,   3,   8,   3,
         7,  7,  7,  5,  4,  -3,  -5,  -3,
         4,  3, 13,  1,  2,   1,  -1,   2,
         3,  5,  8,  4, -5,  -6,  -8, -11,
        -4,  0, -5, -1, -7, -12,  -8, -16,
        -6, -6,  0,  2, -9,  -9, -11,  -3,
        -9,  2,  3, -1, -5, -13,   4, -20,
    ],
    [
         -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
          3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    ],
    [
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
         10,  17,  23,  15,  20,  45,  44,  13,
         -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43,
    ],
];

/// Expands value and piece-square tables into a `[piece][square]` table for all
/// twelve coloured pieces. White reads the table mirrored vertically (`sq ^ 56`)
/// because the tables are written rank 8 first; black reads it directly, which
/// gives the symmetry the tests check.
const fn build_table(value: &[i32; 6], pst: &[[i32; 64]; 6]) -> [[i32; 64]; 12] {
    let mut table = [[0i32; 64]; 12];
    let mut piece = 0;
    while piece < 6 {
        let mut sq = 0;
        while sq < 64 {
            table[piece][sq] = value[piece] + pst[piece][sq ^ 56];
            table[piece + 6][sq] = value[piece] + pst[piece][sq];
            sq += 1;
        }
        piece += 1;
    }
    table
}

/// Material plus piece-square bonus for every coloured piece on every square, middlegame.
static MG_TABLE: [[i32; 64]; 12] = build_table(&MG_VALUE, &MG_PST);
/// Material plus piece-square bonus for every coloured piece on every square, endgame.
static EG_TABLE: [[i32; 64]; 12] = build_table(&EG_VALUE, &EG_PST);

/// Static evaluation of `pos` in centipawns from the side to move's perspective.
///
/// Uses PeSTO tapered piece-square tables plus a small tempo bonus. The function
/// is symmetric: mirroring the board and swapping the colours gives the same
/// score.
pub fn evaluate(pos: &Position) -> i32 {
    let mut mg = 0i32;
    let mut eg = 0i32;
    let mut phase = 0i32;

    for piece_type in PieceType::ALL {
        phase += PHASE_WEIGHT[piece_type.idx()] * pos.piece_bb(piece_type).count() as i32;
        for color in Color::ALL {
            let piece = Piece::new(color, piece_type).idx();
            let sign = if color == Color::White { 1 } else { -1 };
            for sq in pos.pieces(color, piece_type) {
                mg += sign * MG_TABLE[piece][sq.idx()];
                eg += sign * EG_TABLE[piece][sq.idx()];
            }
        }
    }

    let phase = phase.min(TOTAL_PHASE);
    let white_score = (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE;
    let score = if pos.side_to_move() == Color::White {
        white_score
    } else {
        -white_score
    };
    score + TEMPO
}

/// Distance of each file or rank from the centre of the board: 3 on the edge, 0 on the central pair.
const CENTER_DISTANCE: [i32; 8] = [3, 2, 1, 0, 0, 1, 2, 3];
/// Centipawns awarded per step the bare king is pushed from the centre.
const MOP_UP_EDGE: i32 = 40;
/// Centipawns awarded per step the winning king approaches the bare king.
const MOP_UP_CLOSE: i32 = 14;

/// Mop-up evaluation for positions where one side has only a king and the other
/// side has at least a rook or queen. Returns `None` for every other position.
///
/// The score is the stronger side's material (using endgame values) plus a bonus
/// for driving the bare king toward the edge and a bonus for bringing the kings
/// close together. It is returned from the side to move's perspective like
/// [`evaluate`].
///
/// The search calls this before consulting the network, so these endings are
/// never scored by the net.
pub fn bare_king_eval(pos: &Position) -> Option<i32> {
    for strong in Color::ALL {
        let weak = strong.flip();
        let weak_has_only_king = pos.color_bb(weak) == pos.pieces(weak, PieceType::King);
        let heavy =
            (pos.pieces(strong, PieceType::Rook) | pos.pieces(strong, PieceType::Queen)).any();
        if !weak_has_only_king || !heavy {
            continue;
        }

        let mut material = 0i32;
        for piece_type in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            material += EG_VALUE[piece_type.idx()] * pos.pieces(strong, piece_type).count() as i32;
        }

        let weak_king = pos.king_square(weak);
        let strong_king = pos.king_square(strong);
        let edge =
            CENTER_DISTANCE[weak_king.file() as usize] + CENTER_DISTANCE[weak_king.rank() as usize];
        let file_gap = (weak_king.file() as i32 - strong_king.file() as i32).abs();
        let rank_gap = (weak_king.rank() as i32 - strong_king.rank() as i32).abs();
        let close = 14 - (file_gap + rank_gap);

        let score = material + MOP_UP_EDGE * edge + MOP_UP_CLOSE * close;
        return Some(if pos.side_to_move() == strong {
            score + TEMPO
        } else {
            -score + TEMPO
        });
    }
    None
}
