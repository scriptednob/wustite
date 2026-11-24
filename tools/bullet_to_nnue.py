"""Convert a Bullet `quantised.bin` (768x128 SCReLU) into Wustite's WNNU v1 format.

Reads the packed little-endian `i16` weights and biases from the front of
Bullet's binary (ignoring any trailing padding), prepends the 16-byte `WNNU`
v1 header, and widens the output bias from `i16` to `i32`.

Usage:
    python3 tools/bullet_to_nnue.py <quantised.bin> <output.nnue>
"""

import struct
import sys

INPUTS = 768
HIDDEN = 128

src, dst = sys.argv[1], sys.argv[2]
data = open(src, "rb").read()

# Bullet lays out all parameters as contiguous little-endian i16s (with optional trailing padding).
count = INPUTS * HIDDEN + HIDDEN + 2 * HIDDEN + 1
vals = struct.unpack(f"<{count}h", data[: 2 * count])

ft_w = vals[: INPUTS * HIDDEN]
ft_b = vals[INPUTS * HIDDEN : INPUTS * HIDDEN + HIDDEN]
out_w = vals[INPUTS * HIDDEN + HIDDEN : INPUTS * HIDDEN + HIDDEN + 2 * HIDDEN]
out_b = vals[-1]

with open(dst, "wb") as f:
    # Header: magic ("WNNU"), version (1), input size (768), hidden size (128).
    f.write(b"WNNU" + struct.pack("<III", 1, INPUTS, HIDDEN))
    f.write(struct.pack(f"<{len(ft_w) + len(ft_b) + len(out_w)}h", *ft_w, *ft_b, *out_w))
    # WNNU v1 stores the output bias as a 32-bit signed int.
    f.write(struct.pack("<i", out_b))