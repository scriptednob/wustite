"""Convert Wustite datagen files into Bullet's text training format.

Wustite's `datagen` writes `fen;score;result` lines from the side to move's
perspective (`result` is 0, 1, or 2). Bullet expects `fen | score | result`
from White's perspective (`0.0`, `0.5`, or `1.0`), so this flips the score and
result whenever Black is to move and skips any malformed lines.

Usage:
    python3 <script.py> <input.txt> <output.txt>
"""

import sys

src, dst = sys.argv[1], sys.argv[2]

with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for line in f:
        parts = line.strip().split(";")
        if len(parts) != 3:
            continue
        fen, score, result = parts
        score = int(float(score))
        r = int(result)
        # Flip side-to-move labels to White's perspective.
        if fen.split()[1] == "b":
            score = -score
            r = 2 - r
        out.write(f"{fen} | {score} | {r / 2:.1f}\n")