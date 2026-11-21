"""Build a text file of balanced opening FENs for match testing.

Walks 6 or 8 random legal plies from the starting position, skips finished
games and duplicate transpositions, and keeps positions whose depth-7 search
score stays within +/-100 cp.

Usage:
    python3 tools/make_openings.py <engine_binary> <output_path> <count>

Requires python-chess (`pip install chess`). The RNG seed is fixed to 4242 so
re-running against the same engine build produces the same book.
"""

import random, subprocess, sys
import chess

random.seed(4242)
binary, out_path, count = sys.argv[1], sys.argv[2], int(sys.argv[3])

proc = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)


def send(line):
    """Send one UCI command line to the engine and flush it right away."""
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


def wait(prefix):
    """Block until the engine prints a line starting with `prefix`."""
    while True:
        line = proc.stdout.readline().strip()
        if line.startswith(prefix):
            return line


def eval_depth(fen, depth):
    """Search `fen` to `depth` plies and return the final centipawn score."""
    send(f"position fen {fen}")
    send(f"go depth {depth}")
    score = 0
    while True:
        line = proc.stdout.readline().strip()
        if line.startswith("info") and " score cp " in line:
            tok = line.split()
            score = int(tok[tok.index("cp") + 1])
        if line.startswith("bestmove"):
            return score


send("uci")
wait("uciok")
seen = set()
fens = []
while len(fens) < count:
    b = chess.Board()
    for _ in range(random.choice([6, 8])):
        moves = list(b.legal_moves)
        b.push(random.choice(moves))
        if b.is_game_over():
            break
    if b.is_game_over():
        continue
    fen = b.fen()
    # Strip halfmove/fullmove clocks so transpositions don't slip through twice.
    key = fen.rsplit(" ", 2)[0]
    if key in seen:
        continue
    if abs(eval_depth(fen, 7)) <= 100:
        seen.add(key)
        fens.append(fen)

open(out_path, "w").write("\n".join(fens) + "\n")
send("quit")
print(len(fens))
