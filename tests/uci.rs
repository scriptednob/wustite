//! End-to-end UCI protocol tests spawning the compiled `wustite` binary.
//!
//! Covers handshake/options, move-list playback, FEN + depth search, `stop` on
//! `go infinite`, `go perft`, and `go ponder` / `ponderhit` timing.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use wustite::Position;
use wustite::movegen::{legal_moves, parse_move};

struct Session {
    child: std::process::Child,
    reader: BufReader<std::process::ChildStdout>,
}

impl Session {
    fn start() -> Session {
        let mut child = Command::new(env!("CARGO_BIN_EXE_wustite"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("engine starts");
        let stdout = child.stdout.take().expect("stdout");
        Session {
            child,
            reader: BufReader::new(stdout),
        }
    }

    fn send(&mut self, line: &str) {
        let stdin = self.child.stdin.as_mut().expect("stdin");
        writeln!(stdin, "{line}").expect("write");
        stdin.flush().expect("flush");
    }

    fn read_until(&mut self, prefix: &str) -> Vec<String> {
        let mut lines = Vec::new();
        loop {
            let mut line = String::new();
            let n = self.reader.read_line(&mut line).expect("read");
            assert!(
                n > 0,
                "engine closed before sending {prefix:?}; got {lines:?}"
            );
            let line = line.trim().to_string();
            let done = line.starts_with(prefix);
            lines.push(line);
            if done {
                return lines;
            }
        }
    }

    fn quit(mut self) {
        self.send("quit");
        let status = self.child.wait().expect("exit");
        assert!(status.success());
    }
}

#[test]
fn handshake_and_options() {
    let mut engine = Session::start();
    engine.send("uci");
    let lines = engine.read_until("uciok");
    assert!(lines.iter().any(|l| l.starts_with("id name Wustite")));
    assert!(lines.iter().any(|l| l.contains("option name Hash")));
    assert!(lines.iter().any(|l| l.contains("option name Threads")));
    engine.send("setoption name Hash value 32");
    engine.send("setoption name Threads value 2");
    engine.send("isready");
    engine.read_until("readyok");
    engine.quit();
}

#[test]
fn plays_a_legal_move_from_a_move_list() {
    let mut engine = Session::start();
    engine.send("uci");
    engine.read_until("uciok");
    engine.send("ucinewgame");
    engine.send("position startpos moves e2e4 e7e5 g1f3 b8c6");
    engine.send("go movetime 200");
    let lines = engine.read_until("bestmove");
    let best = lines
        .last()
        .and_then(|l| l.split_whitespace().nth(1))
        .expect("bestmove token")
        .to_string();

    let mut pos = Position::startpos();
    for text in ["e2e4", "e7e5", "g1f3", "b8c6"] {
        pos = pos.make_move(parse_move(&pos, text).expect("legal"));
    }
    assert!(
        legal_moves(&pos).iter().any(|m| m.to_string() == best),
        "illegal bestmove {best}"
    );
    assert!(lines.iter().any(|l| l.starts_with("info depth")));
    engine.quit();
}

#[test]
fn fen_position_and_depth_limit() {
    let mut engine = Session::start();
    engine.send("position fen 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1");
    engine.send("go depth 6");
    let lines = engine.read_until("bestmove");
    assert_eq!(lines.last().expect("line"), "bestmove a1a8");
    assert!(lines.iter().any(|l| l.contains("score mate 1")));
    engine.quit();
}

#[test]
fn stop_ends_an_infinite_search() {
    let mut engine = Session::start();
    engine.send("position startpos");
    engine.send("go infinite");
    engine.read_until("info depth 3");
    engine.send("stop");
    let lines = engine.read_until("bestmove");
    assert!(lines.last().expect("line").starts_with("bestmove "));
    engine.quit();
}

#[test]
fn go_perft_matches_known_count() {
    let mut engine = Session::start();
    engine.send("position startpos");
    engine.send("go perft 4");
    let lines = engine.read_until("Nodes searched");
    assert_eq!(lines.last().expect("line"), "Nodes searched: 197281");
    engine.quit();
}

struct Channel {
    child: std::process::Child,
    lines: mpsc::Receiver<String>,
}

impl Channel {
    fn start() -> Channel {
        let mut child = Command::new(env!("CARGO_BIN_EXE_wustite"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("engine starts");
        let stdout = child.stdout.take().expect("stdout");
        let (sender, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        Channel { child, lines }
    }

    fn send(&mut self, line: &str) {
        let stdin = self.child.stdin.as_mut().expect("stdin");
        writeln!(stdin, "{line}").expect("write");
        stdin.flush().expect("flush");
    }

    fn wait_for(&self, prefix: &str, timeout: Duration) -> Option<String> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            match self.lines.recv_timeout(remaining) {
                Ok(line) if line.starts_with(prefix) => return Some(line),
                Ok(_) => {}
                Err(_) => return None,
            }
        }
    }

    fn quit(mut self) {
        self.send("quit");
        let status = self.child.wait().expect("exit");
        assert!(status.success());
    }
}

#[test]
fn ponder_search_waits_for_ponderhit_then_answers() {
    let mut engine = Channel::start();
    engine.send("uci");
    assert!(engine.wait_for("uciok", Duration::from_secs(5)).is_some());
    engine.send("position startpos moves e2e4 e7e5");
    engine.send("go ponder wtime 300 btime 300 winc 0 binc 0");
    assert!(
        engine
            .wait_for("bestmove", Duration::from_millis(1500))
            .is_none(),
        "bestmove sent while pondering"
    );
    engine.send("ponderhit");
    let reply = engine
        .wait_for("bestmove", Duration::from_secs(5))
        .expect("bestmove after ponderhit");
    assert!(reply.split_whitespace().nth(1).is_some());
    engine.quit();
}

#[test]
fn ponder_search_ends_on_stop() {
    let mut engine = Channel::start();
    engine.send("position startpos");
    engine.send("go ponder wtime 60000 btime 60000");
    assert!(
        engine
            .wait_for("info depth 3", Duration::from_secs(5))
            .is_some()
    );
    engine.send("stop");
    assert!(
        engine
            .wait_for("bestmove", Duration::from_secs(5))
            .is_some()
    );
    engine.quit();
}

#[test]
fn ponderhit_time_is_counted_from_the_hit() {
    let mut engine = Channel::start();
    engine.send("position startpos");
    engine.send("go ponder wtime 400 btime 400 winc 0 binc 0");
    std::thread::sleep(Duration::from_millis(900));
    let hit = std::time::Instant::now();
    engine.send("ponderhit");
    assert!(
        engine
            .wait_for("bestmove", Duration::from_secs(5))
            .is_some()
    );
    assert!(
        hit.elapsed() < Duration::from_millis(900),
        "{:?}",
        hit.elapsed()
    );
    engine.quit();
}
