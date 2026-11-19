import argparse, math, random, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import chess


class Engine:
    def __init__(self, path, options, overhead):
        self.proc = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("uci")
        self.read_until("uciok")
        self.send(f"setoption name Move Overhead value {overhead}")
        for opt in options:
            self.send(f"setoption name {opt[0]} value {opt[1]}")
        self.send("isready")
        self.read_until("readyok")

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefix):
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died, last lines: " + repr(lines[-5:]))
            line = line.strip()
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def newgame(self):
        self.send("ucinewgame")
        self.send("isready")
        self.read_until("readyok")

    def go(self, start_fen, moves, go_cmd):
        self.send(f"position fen {start_fen} moves {' '.join(moves)}" if moves else f"position fen {start_fen}")
        t0 = time.perf_counter()
        self.send(go_cmd)
        lines = self.read_until("bestmove")
        elapsed = time.perf_counter() - t0
        score = 0
        depth = 0
        for line in lines:
            if line.startswith("info") and " score " in line:
                tok = line.split()
                i = tok.index("score")
                if tok[i + 1] == "cp":
                    score = int(tok[i + 2])
                elif tok[i + 1] == "mate":
                    m = int(tok[i + 2])
                    score = 10000 if m > 0 else -10000
                if "depth" in tok:
                    depth = int(tok[tok.index("depth") + 1])
        best = lines[-1].split()[1]
        return best, score, elapsed, depth

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def play_game(path_white, path_black, opts_white, opts_black, opening, args, game_id):
    ew = Engine(path_white, opts_white, args.overhead)
    eb = Engine(path_black, opts_black, args.overhead)
    try:
        ew.newgame()
        eb.newgame()
        board = chess.Board(opening)
        moves = []
        history = []
        clocks = [args.base * 1000.0, args.base * 1000.0]
        inc = args.inc * 1000.0
        depths = []
        while True:
            if board.is_checkmate():
                return ("black" if board.turn == chess.WHITE else "white"), "mate", depths
            if (board.is_stalemate() or board.is_insufficient_material()
                    or board.can_claim_threefold_repetition() or board.halfmove_clock >= 100
                    or board.is_fivefold_repetition()):
                return "draw", "rule", depths
            if len(moves) >= args.max_plies:
                return "draw", "maxplies", depths
            side = 0 if board.turn == chess.WHITE else 1
            eng = ew if side == 0 else eb
            if args.movetime:
                cmd = f"go movetime {args.movetime}"
            else:
                cmd = f"go wtime {int(clocks[0])} btime {int(clocks[1])} winc {int(inc)} binc {int(inc)}"
            best, score, elapsed, depth = eng.go(opening, moves, cmd)
            depths.append(depth)
            if not args.movetime:
                clocks[side] -= elapsed * 1000.0
                if clocks[side] < 0:
                    return ("black" if side == 0 else "white"), "time", depths
                clocks[side] += inc
            try:
                mv = chess.Move.from_uci(best)
            except ValueError:
                return ("black" if side == 0 else "white"), "badmove", depths
            if mv not in board.legal_moves:
                return ("black" if side == 0 else "white"), "illegal:" + best, depths
            board.push(mv)
            moves.append(best)
            history.append((side, score))
            if len(history) >= 6:
                last = history[-6:]
                win_side = None
                mine = [s for sd, s in last if sd == last[-1][0]]
                theirs = [s for sd, s in last if sd != last[-1][0]]
                if all(s >= 900 for s in mine) and all(s <= -900 for s in theirs):
                    win_side = last[-1][0]
                elif all(s <= -900 for s in mine) and all(s >= 900 for s in theirs):
                    win_side = 1 - last[-1][0]
                if win_side is not None:
                    return ("white" if win_side == 0 else "black"), "adjudicated", depths
    finally:
        ew.close()
        eb.close()


def llr(w, d, l, elo0, elo1):
    n = w + d + l
    if n == 0:
        return 0.0
    mean = (w + 0.5 * d) / n
    var = (w * (1 - mean) ** 2 + d * (0.5 - mean) ** 2 + l * (0 - mean) ** 2) / n
    if var <= 0:
        return 0.0
    s0 = 1 / (1 + 10 ** (-elo0 / 400))
    s1 = 1 / (1 + 10 ** (-elo1 / 400))
    return n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)


def elo_estimate(w, d, l):
    n = w + d + l
    if n == 0:
        return 0.0, 0.0
    mean = (w + 0.5 * d) / n
    var = (w * (1 - mean) ** 2 + d * (0.5 - mean) ** 2 + l * (0 - mean) ** 2) / n
    sd = math.sqrt(var / n)

    def to_elo(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return -400 * math.log10(1 / p - 1)

    return to_elo(mean), (to_elo(mean + 1.96 * sd) - to_elo(mean - 1.96 * sd)) / 2


def parse_opts(text):
    out = []
    for item in [t for t in text.split(",") if t]:
        k, v = item.split("=")
        out.append((k.strip(), v.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--bin-b", default="")
    ap.add_argument("--a", default="")
    ap.add_argument("--b", default="")
    ap.add_argument("--openings", required=True)
    ap.add_argument("--movetime", type=int, default=0)
    ap.add_argument("--base", type=float, default=5.0)
    ap.add_argument("--inc", type=float, default=0.05)
    ap.add_argument("--overhead", type=int, default=5)
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=10.0)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--no-sprt", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    openings = [l.strip() for l in open(args.openings) if l.strip()]
    opts_a = parse_opts(args.a)
    opts_b = parse_opts(args.b)
    lower, upper = math.log(0.05 / 0.95), math.log(0.95 / 0.05)

    w = d = l = 0
    losses_on_time = 0
    other_failures = []
    done = 0
    lock = threading.Lock()
    stop = threading.Event()
    ends = {}
    crossed = []

    def job(i):
        if stop.is_set():
            return None
        opening = openings[(i // 2) % len(openings)]
        a_white = (i % 2 == 0)
        ow, ob = (opts_a, opts_b) if a_white else (opts_b, opts_a)
        bin_b = args.bin_b or args.bin
        pw, pb = (args.bin, bin_b) if a_white else (bin_b, args.bin)
        result, why, depths = play_game(pw, pb, ow, ob, opening, args, i)
        return i, a_white, result, why

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(job, i) for i in range(args.games)]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            i, a_white, result, why = res
            with lock:
                done += 1
                ends[why.split(":")[0]] = ends.get(why.split(":")[0], 0) + 1
                if why.startswith("illegal") or why == "badmove":
                    other_failures.append((i, why))
                if result == "draw":
                    d += 1
                elif (result == "white") == a_white:
                    w += 1
                else:
                    l += 1
                if why == "time":
                    losses_on_time += 1
                value = llr(w, d, l, args.elo0, args.elo1)
                if done % 20 == 0 or done == args.games:
                    e, ci = elo_estimate(w, d, l)
                    print(f"[{args.label}] games {done} W{w} D{d} L{l} elo {e:+.1f} +-{ci:.1f} llr {value:+.2f} ({time.time()-t0:.0f}s)", flush=True)
                if not args.no_sprt and (value >= upper or value <= lower) and not crossed:
                    crossed.append('H1' if value >= upper else 'H0')
                    stop.set()
    e, ci = elo_estimate(w, d, l)
    value = llr(w, d, l, args.elo0, args.elo1)
    verdict = "inconclusive"
    if crossed and crossed[0] == 'H1':
        verdict = "H1 accepted (gain)"
    elif crossed and crossed[0] == 'H0':
        verdict = "H0 accepted (no gain)"
    print(f"[{args.label}] FINAL games {w+d+l} W{w} D{d} L{l} elo {e:+.1f} +-{ci:.1f} llr {value:+.2f} verdict: {verdict} timeouts {losses_on_time} failures {other_failures} ends {ends}", flush=True)


if __name__ == "__main__":
    main()
