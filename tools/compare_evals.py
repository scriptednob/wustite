"""Compare two `.nnue` networks against datagen search scores on sample positions.

Reads the first `n` `fen;score;wdl` lines from a datagen file, batches UCI
`eval` commands through the engine binary for each network, and prints the
Pearson correlation and mean absolute centipawn error against the datagen
scores.

Usage:
    python3 tools/compare_evals.py <wustite_bin> <net_a.nnue> <net_b.nnue> <data.txt> <n>
"""

import statistics
import subprocess
import sys

exe, net_a, net_b, data, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])

rows = []
with open(data, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= n:
            break
        fen, score, _ = line.rstrip("\n").split(";")
        rows.append((fen, int(float(score))))


def evals(net):
    """Run static `eval` for every sampled FEN using `net` in one engine pass."""
    cmds = ["uci", f"setoption name EvalFile value {net}"]
    for fen, _ in rows:
        cmds += [f"position fen {fen}", "eval"]
    cmds.append("quit")
    out = subprocess.run([exe], input="\n".join(cmds) + "\n", capture_output=True, text=True).stdout
    return [int(line.split()[-1]) for line in out.splitlines() if line.startswith("static eval")]


scores = [s for _, s in rows]
for net in (net_a, net_b):
    vals = evals(net)
    print(
        net,
        "correlation",
        round(statistics.correlation(vals, scores), 3),
        "mean abs diff",
        round(statistics.mean(abs(v - s) for v, s in zip(vals, scores))),
    )