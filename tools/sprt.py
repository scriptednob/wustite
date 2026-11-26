"""Build two versions of Wustite and run a fastchess SPRT between them. Run with --help for options."""

import argparse
import hashlib
import io
import os
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "testing"
EXE = ".exe" if os.name == "nt" else ""
BOOK_NAME = "UHO_Lichess_4852_v1.epd"
BOOK_URL = f"https://github.com/official-stockfish/books/raw/37751b08db745aa02babfa74056dd1168b9da803/{BOOK_NAME}.zip"
BOOK_SHA256 = "4e298f11e8acfa106babe02968f2e61582145e7874c59284690b20b9650e0e07"
PUBLIC_OPTIONS = {"usennue", "evalfile"}


def parse_options(text):
    options = {}
    for item in filter(None, text.split(",")):
        name, sep, value = item.partition("=")
        if not sep:
            sys.exit(f"bad option '{item}', expected name=value")
        options[name.strip()] = value.strip()
    return options


def find_fastchess(given):
    for candidate in (given, WORK / f"fastchess{EXE}", shutil.which("fastchess")):
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    sys.exit(
        "fastchess not found. Download it from https://github.com/Disservin/fastchess/releases "
        f"and put fastchess{EXE} in {WORK}, or pass --fastchess PATH."
    )


def find_book(given):
    if given:
        path = Path(given).resolve()
        if not path.is_file():
            sys.exit(f"book not found: {path}")
        return path
    path = WORK / BOOK_NAME
    if path.is_file():
        return path
    print(f"downloading {BOOK_NAME} ...", flush=True)
    with urllib.request.urlopen(BOOK_URL) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != BOOK_SHA256:
        sys.exit("opening book download failed its checksum")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extract(BOOK_NAME, WORK)
    return path


def default_rustflags():
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        text = cpuinfo.read_text(errors="ignore")
        zen_with_slow_pext = "AuthenticAMD" in text and "cpu family\t: 23\n" in text
    else:
        zen_with_slow_pext = "AuthenticAMD" in cpu and "Family 23 " in cpu
    flags = "-C target-cpu=native"
    if zen_with_slow_pext:
        flags += " -C target-feature=-bmi2"
    return flags


def build(source, name, tune):
    target = WORK / f"target-{name}"
    command = ["cargo", "build", "--release", "--bin", "wustite", "--target-dir", str(target)]
    if tune:
        command += ["--features", "tune"]
    env = dict(os.environ)
    env.setdefault("RUSTFLAGS", default_rustflags())
    print(f"building {name} (RUSTFLAGS={env['RUSTFLAGS']}) ...", flush=True)
    subprocess.run(command, cwd=source, env=env, check=True)
    output = WORK / "bin" / f"{name}{EXE}"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target / "release" / f"wustite{EXE}", output)
    return output


def export_ref(ref):
    destination = WORK / "base-src"
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", "--format=tar", ref], cwd=ROOT, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(destination, filter="data")
        else:
            tar.extractall(destination)
    return destination


def resolve_options(binary, options):
    output = subprocess.run([str(binary)], input="uci\nquit\n", capture_output=True, text=True, check=True).stdout
    listed = {}
    for line in output.splitlines():
        if line.startswith("option name "):
            name = line[len("option name "):].split(" type ")[0]
            listed[name.lower()] = name
    missing = [name for name in options if name.lower() not in listed]
    if missing:
        sys.exit(f"{binary.name} does not list these options, so fastchess would ignore them: {', '.join(missing)}")
    return {listed[name.lower()]: value for name, value in options.items()}


def bench(binary):
    output = subprocess.run([str(binary), "bench", "10"], capture_output=True, text=True, check=True).stdout
    return next(line for line in output.splitlines() if line.startswith("nodes "))


def main():
    ap = argparse.ArgumentParser(description="SPRT the working tree against a git ref using fastchess.")
    ap.add_argument("--base", default="HEAD", help="git ref to test against (default: HEAD)")
    ap.add_argument("--new-options", default="", help="UCI options for the new side, e.g. histlmr=1")
    ap.add_argument("--base-options", default="", help="UCI options for the base side")
    ap.add_argument("--tc", default="8+0.08", help="time control in seconds+increment")
    ap.add_argument("--concurrency", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel games (default: cores minus one)")
    ap.add_argument("--hash", type=int, default=16, help="hash per engine in MiB")
    ap.add_argument("--overhead", type=int, default=10, help="Move Overhead in ms")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=10.0)
    ap.add_argument("--rounds", type=int, default=20000, help="maximum game pairs")
    ap.add_argument("--fastchess", default="", help="path to the fastchess executable")
    ap.add_argument("--book", default="", help="path to an EPD opening book")
    ap.add_argument("--build-only", action="store_true", help="build and check both sides, then stop")
    args = ap.parse_args()

    WORK.mkdir(exist_ok=True)
    new_options = parse_options(args.new_options)
    base_options = parse_options(args.base_options)

    def needs_tune(options):
        return any(name.lower() not in PUBLIC_OPTIONS for name in options)

    new = build(ROOT, "new", needs_tune(new_options))
    base = build(export_ref(args.base), "base", needs_tune(base_options))
    new_options = resolve_options(new, new_options)
    base_options = resolve_options(base, base_options)
    print(f"new  bench: {bench(new)}")
    print(f"base bench: {bench(base)} ({args.base})")
    if args.build_only:
        return

    fastchess = find_fastchess(args.fastchess)
    book = find_book(args.book)
    games = WORK / "games"
    games.mkdir(exist_ok=True)

    def engine(binary, name, options):
        return ["-engine", f"cmd={binary}", f"name={name}"] + [f"option.{k}={v}" for k, v in options.items()]

    command = [
        fastchess,
        *engine(new, "new", new_options),
        *engine(base, "base", base_options),
        "-each", "proto=uci", f"tc={args.tc}", f"option.Hash={args.hash}", "option.Threads=1",
        f"option.Move Overhead={args.overhead}",
        "-openings", f"file={book}", "format=epd", "order=random",
        "-srand", str(random.randrange(1, 2**31)),
        "-rounds", str(args.rounds), "-repeat",
        "-concurrency", str(args.concurrency),
        "-sprt", f"elo0={args.elo0}", f"elo1={args.elo1}", "alpha=0.05", "beta=0.05",
        "-resign", "movecount=3", "score=600",
        "-draw", "movenumber=34", "movecount=8", "score=20",
        "-ratinginterval", "10",
        "-recover",
        "-pgnout", f"file={games / time.strftime('%Y%m%d-%H%M%S.pgn')}",
    ]
    try:
        sys.exit(subprocess.run(command, cwd=WORK).returncode)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
