//! Pure-Rust NNUE trainer for Wustite's `(768 -> 128) x 2 -> 1` SCReLU network.
//!
//! Reads `fen;score;wdl` files produced by `examples/datagen.rs`, trains float
//! weights with Adam (exponential learning-rate decay) against MSE on sigmoid
//! targets blended by `--lambda`, and writes both a quantized `.nnue` network
//! and a raw float checkpoint (`<out>.params`) after every epoch.
//!
//! At the end of training, prints the mean and worst-case centipawn difference
//! between the float model and the quantized `i16` network on a validation sample.
//!
//! Example:
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --example train -- \
//!     --data data.txt --out net.nnue --epochs 150 --threads 4
//! ```

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use wustite::Position;
use wustite::nnue::{self, HIDDEN, INPUTS, QA, QB, SCALE};

const MAX_PIECES: usize = 32;
const WEIGHT_LIMIT: f32 = 1.98;
const OUTPUT_WEIGHT_LIMIT: f32 = 127.0 / QB as f32;
const LABEL_SCALE: f32 = 400.0;

/// Minimal XorShift64* RNG for weight init, train/val split, and epoch shuffles.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Pre-extracted active feature indices for both perspectives plus the blended target.
#[derive(Clone, Copy)]
struct Sample {
    us: [u16; MAX_PIECES],
    them: [u16; MAX_PIECES],
    count: u8,
    target: f32,
}

/// Float network weights trained by Adam before quantizing into `WNNU` v1 format.
#[derive(Clone)]
struct Params {
    w: Vec<f32>,
    b: Vec<f32>,
    ow: Vec<f32>,
    ob: f32,
}

impl Params {
    fn zeros() -> Params {
        Params {
            w: vec![0.0; INPUTS * HIDDEN],
            b: vec![0.0; HIDDEN],
            ow: vec![0.0; 2 * HIDDEN],
            ob: 0.0,
        }
    }

    fn random(rng: &mut Rng) -> Params {
        let mut p = Params::zeros();
        for w in p.w.iter_mut() {
            *w = (rng.unit() * 2.0 - 1.0) * 0.1;
        }
        for b in p.b.iter_mut() {
            *b = 0.1;
        }
        for w in p.ow.iter_mut() {
            *w = (rng.unit() * 2.0 - 1.0) * 0.1;
        }
        p
    }

    /// Keep float weights inside the representable quantization range (especially `+-OUTPUT_WEIGHT_LIMIT`).
    fn clamp(&mut self) {
        for w in self.w.iter_mut().chain(self.b.iter_mut()) {
            *w = w.clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
        }
        for w in self.ow.iter_mut() {
            *w = w.clamp(-OUTPUT_WEIGHT_LIMIT, OUTPUT_WEIGHT_LIMIT);
        }
    }

    /// Quantize float weights (`QA`, `QB`) and serialize them as a `WNNU` v1 `.nnue` file.
    fn to_bytes(&self) -> Vec<u8> {
        let ft_weights: Vec<i16> = self
            .w
            .iter()
            .map(|w| (w * QA as f32).round().clamp(-32767.0, 32767.0) as i16)
            .collect();
        let ft_bias: Vec<i16> = self
            .b
            .iter()
            .map(|w| (w * QA as f32).round().clamp(-32767.0, 32767.0) as i16)
            .collect();
        let out_weights: Vec<i16> = self
            .ow
            .iter()
            .map(|w| (w * QB as f32).round().clamp(-127.0, 127.0) as i16)
            .collect();
        let out_bias = (self.ob * (QA * QB) as f32).round() as i32;
        nnue::serialize(&ft_weights, &ft_bias, &out_weights, out_bias)
    }

    /// Dump raw little-endian `f32` weights so training can be resumed with `--resume`.
    fn save_raw(&self, path: &str) -> std::io::Result<()> {
        let mut bytes = Vec::with_capacity((self.w.len() + self.b.len() + self.ow.len() + 1) * 4);
        for v in self.w.iter().chain(&self.b).chain(&self.ow) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&self.ob.to_le_bytes());
        fs::write(path, bytes)
    }

    /// Load a `.params` float checkpoint previously saved by [`Params::save_raw`].
    fn load_raw(path: &str) -> Result<Params, String> {
        let bytes = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let expected = (INPUTS * HIDDEN + HIDDEN + 2 * HIDDEN + 1) * 4;
        if bytes.len() != expected {
            return Err(format!(
                "{path}: expected {expected} bytes, found {}",
                bytes.len()
            ));
        }
        let mut values = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        let mut p = Params::zeros();
        for v in p.w.iter_mut().chain(p.b.iter_mut()).chain(p.ow.iter_mut()) {
            *v = values.next().unwrap_or(0.0);
        }
        p.ob = values.next().unwrap_or(0.0);
        Ok(p)
    }
}

/// Accumulated gradients and MSE loss over a mini-batch shard.
struct Grad {
    w: Vec<f32>,
    b: Vec<f32>,
    ow: Vec<f32>,
    ob: f32,
    loss: f64,
    count: usize,
}

impl Grad {
    fn zeros() -> Grad {
        Grad {
            w: vec![0.0; INPUTS * HIDDEN],
            b: vec![0.0; HIDDEN],
            ow: vec![0.0; 2 * HIDDEN],
            ob: 0.0,
            loss: 0.0,
            count: 0,
        }
    }

    fn merge(&mut self, other: &Grad) {
        for (a, b) in self.w.iter_mut().zip(&other.w) {
            *a += *b;
        }
        for (a, b) in self.b.iter_mut().zip(&other.b) {
            *a += *b;
        }
        for (a, b) in self.ow.iter_mut().zip(&other.ow) {
            *a += *b;
        }
        self.ob += other.ob;
        self.loss += other.loss;
        self.count += other.count;
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn accumulate_side(params: &Params, features: &[u16], acc: &mut [f32; HIDDEN]) {
    acc.copy_from_slice(&params.b);
    for &f in features {
        let column = &params.w[f as usize * HIDDEN..(f as usize + 1) * HIDDEN];
        for (a, w) in acc.iter_mut().zip(column) {
            *a += *w;
        }
    }
}

/// Run a float forward pass for one position (returns raw logit before sigmoid).
fn forward(params: &Params, sample: &Sample) -> f32 {
    let n = sample.count as usize;
    let mut us = [0f32; HIDDEN];
    let mut them = [0f32; HIDDEN];
    accumulate_side(params, &sample.us[..n], &mut us);
    accumulate_side(params, &sample.them[..n], &mut them);
    let mut out = params.ob;
    for h in 0..HIDDEN {
        let a = us[h].clamp(0.0, 1.0);
        let b = them[h].clamp(0.0, 1.0);
        out += params.ow[h] * a * a + params.ow[HIDDEN + h] * b * b;
    }
    out
}

/// Compute MSE loss and accumulate analytical gradients for one sample.
fn backward(params: &Params, sample: &Sample, grad: &mut Grad) {
    let n = sample.count as usize;
    let mut us = [0f32; HIDDEN];
    let mut them = [0f32; HIDDEN];
    accumulate_side(params, &sample.us[..n], &mut us);
    accumulate_side(params, &sample.them[..n], &mut them);

    let mut out = params.ob;
    for h in 0..HIDDEN {
        let a = us[h].clamp(0.0, 1.0);
        let b = them[h].clamp(0.0, 1.0);
        out += params.ow[h] * a * a + params.ow[HIDDEN + h] * b * b;
    }
    let p = sigmoid(out);
    let error = p - sample.target;
    grad.loss += (error * error) as f64;
    grad.count += 1;
    let g = 2.0 * error * p * (1.0 - p);
    grad.ob += g;

    let mut d_us = [0f32; HIDDEN];
    let mut d_them = [0f32; HIDDEN];
    for h in 0..HIDDEN {
        let au = us[h].clamp(0.0, 1.0);
        let at = them[h].clamp(0.0, 1.0);
        grad.ow[h] += g * au * au;
        grad.ow[HIDDEN + h] += g * at * at;
        if us[h] > 0.0 && us[h] < 1.0 {
            d_us[h] = g * params.ow[h] * 2.0 * au;
        }
        if them[h] > 0.0 && them[h] < 1.0 {
            d_them[h] = g * params.ow[HIDDEN + h] * 2.0 * at;
        }
    }
    for h in 0..HIDDEN {
        grad.b[h] += d_us[h] + d_them[h];
    }
    for &f in &sample.us[..n] {
        let column = &mut grad.w[f as usize * HIDDEN..(f as usize + 1) * HIDDEN];
        for (c, d) in column.iter_mut().zip(&d_us) {
            *c += *d;
        }
    }
    for &f in &sample.them[..n] {
        let column = &mut grad.w[f as usize * HIDDEN..(f as usize + 1) * HIDDEN];
        for (c, d) in column.iter_mut().zip(&d_them) {
            *c += *d;
        }
    }
}

/// Standard Adam optimizer state (first/second moments and step counter).
struct Adam {
    m: Params,
    v: Params,
    step: i32,
}

impl Adam {
    fn new() -> Adam {
        Adam {
            m: Params::zeros(),
            v: Params::zeros(),
            step: 0,
        }
    }

    fn apply(&mut self, params: &mut Params, grad: &Grad, lr: f32, scale: f32) {
        const BETA1: f32 = 0.9;
        const BETA2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        self.step += 1;
        let c1 = 1.0 - BETA1.powi(self.step);
        let c2 = 1.0 - BETA2.powi(self.step);
        let step = |p: &mut f32, g: f32, m: &mut f32, v: &mut f32| {
            let g = g * scale;
            *m = BETA1 * *m + (1.0 - BETA1) * g;
            *v = BETA2 * *v + (1.0 - BETA2) * g * g;
            *p -= lr * (*m / c1) / ((*v / c2).sqrt() + EPS);
        };
        for i in 0..params.w.len() {
            step(
                &mut params.w[i],
                grad.w[i],
                &mut self.m.w[i],
                &mut self.v.w[i],
            );
        }
        for i in 0..params.b.len() {
            step(
                &mut params.b[i],
                grad.b[i],
                &mut self.m.b[i],
                &mut self.v.b[i],
            );
        }
        for i in 0..params.ow.len() {
            step(
                &mut params.ow[i],
                grad.ow[i],
                &mut self.m.ow[i],
                &mut self.v.ow[i],
            );
        }
        step(&mut params.ob, grad.ob, &mut self.m.ob, &mut self.v.ob);
        params.clamp();
    }
}

/// Extract active NNUE feature indices from `pos` for both perspectives.
fn build_sample(pos: &Position, target: f32) -> Sample {
    let stm = pos.side_to_move();
    let mut sample = Sample {
        us: [0; MAX_PIECES],
        them: [0; MAX_PIECES],
        count: 0,
        target,
    };
    let mut n = 0;
    for sq in pos.occupied() {
        if let Some(piece) = pos.piece_on(sq) {
            if n < MAX_PIECES {
                sample.us[n] = nnue::feature_index(stm, piece, sq) as u16;
                sample.them[n] = nnue::feature_index(stm.flip(), piece, sq) as u16;
                n += 1;
            }
        }
    }
    sample.count = n as u8;
    sample
}

struct Line<'a> {
    fen: &'a str,
    score: f32,
    wdl: f32,
}

/// Parse one `fen;score;wdl` line, normalizing `wdl` (`0/1/2`) to `0.0/0.5/1.0`.
fn parse_line(line: &str) -> Option<Line<'_>> {
    let mut parts = line.split(';');
    let fen = parts.next()?;
    let score: f32 = parts.next()?.trim().parse().ok()?;
    let wdl: f32 = parts.next()?.trim().parse::<f32>().ok()? / 2.0;
    Some(Line { fen, score, wdl })
}

/// Blend search-score sigmoid (`lambda`) with game outcome (`1 - lambda`).
fn target_of(line: &Line, lambda: f32) -> f32 {
    lambda * sigmoid(line.score / LABEL_SCALE) + (1.0 - lambda) * line.wdl
}

struct Settings {
    data: Vec<String>,
    out: String,
    epochs: usize,
    batch: usize,
    lr: f32,
    lr_final: f32,
    lambda: f32,
    threads: usize,
    seed: u64,
    limit: usize,
    validation: f32,
    resume: Option<String>,
}

/// Parse CLI flags (`--data`, `--out`, `--epochs`, `--batch`, `--lr`, `--lr-final`, `--lambda`, `--threads`, `--seed`, `--limit`, `--validation`, `--resume`).
fn parse_settings() -> Result<Settings, String> {
    let mut s = Settings {
        data: Vec::new(),
        out: String::from("net.nnue"),
        epochs: 150,
        batch: 16384,
        lr: 0.008,
        lr_final: 0.0001,
        lambda: 1.0,
        threads: 2,
        seed: 1,
        limit: usize::MAX,
        validation: 0.02,
        resume: None,
    };
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {}", args[i]))?;
        match args[i].as_str() {
            "--data" => s.data = value.split(',').map(str::to_string).collect(),
            "--out" => s.out = value.clone(),
            "--epochs" => s.epochs = value.parse().map_err(|_| "bad --epochs")?,
            "--batch" => s.batch = value.parse().map_err(|_| "bad --batch")?,
            "--lr" => s.lr = value.parse().map_err(|_| "bad --lr")?,
            "--lr-final" => s.lr_final = value.parse().map_err(|_| "bad --lr-final")?,
            "--lambda" => s.lambda = value.parse().map_err(|_| "bad --lambda")?,
            "--threads" => s.threads = value.parse().map_err(|_| "bad --threads")?,
            "--seed" => s.seed = value.parse().map_err(|_| "bad --seed")?,
            "--limit" => s.limit = value.parse().map_err(|_| "bad --limit")?,
            "--validation" => s.validation = value.parse().map_err(|_| "bad --validation")?,
            "--resume" => s.resume = Some(value.clone()),
            other => return Err(format!("unknown option {other}")),
        }
        i += 2;
    }
    if s.data.is_empty() {
        return Err(String::from("--data is required"));
    }
    s.threads = s.threads.max(1);
    s.batch = s.batch.max(1);
    Ok(s)
}

/// Compute mean squared error across `samples` in parallel.
fn evaluate_set(params: &Params, samples: &[Sample], threads: usize) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let chunk = samples.len().div_ceil(threads);
    let total: f64 = std::thread::scope(|scope| {
        let handles: Vec<_> = samples
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    let mut sum = 0.0f64;
                    for s in part {
                        let e = sigmoid(forward(params, s)) - s.target;
                        sum += (e * e) as f64;
                    }
                    sum
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or(0.0)).sum()
    });
    total / samples.len() as f64
}

/// Compare quantized integer inference against the float model on held-out validation FENs.
fn quantization_report(params: &Params, fens: &[(String, f32)]) {
    let Ok(net) = nnue::Network::from_bytes(&params.to_bytes()) else {
        eprintln!("exported network failed to load");
        return;
    };
    let mut diff_sum = 0.0f64;
    let mut worst = 0.0f64;
    let mut counted = 0usize;
    for (fen, _) in fens {
        let Ok(pos) = Position::from_fen(fen) else {
            continue;
        };
        let sample = build_sample(&pos, 0.0);
        let float_cp = forward(params, &sample) as f64 * SCALE as f64;
        let mut acc = nnue::Accumulator::default();
        nnue::refresh(&net, &mut acc, &pos);
        let int_cp = nnue::evaluate(&net, &acc, pos.side_to_move()) as f64;
        let diff = (float_cp - int_cp).abs();
        diff_sum += diff;
        worst = worst.max(diff);
        counted += 1;
    }
    if counted > 0 {
        eprintln!(
            "quantization: mean |float - int| = {:.2} cp, worst {:.1} cp over {counted} positions",
            diff_sum / counted as f64,
            worst
        );
    }
}

fn main() -> ExitCode {
    let settings = match parse_settings() {
        Ok(s) => s,
        Err(message) => {
            eprintln!("{message}");
            eprintln!(
                "usage: train --data FILE[,FILE] [--out NET] [--epochs N] [--batch N] [--lr X] [--lr-final X] [--lambda X] [--threads N] [--limit N] [--validation X] [--resume RAW]"
            );
            return ExitCode::from(2);
        }
    };

    let started = Instant::now();
    let mut texts = Vec::new();
    for path in &settings.data {
        match fs::read_to_string(path) {
            Ok(text) => texts.push(text),
            Err(error) => {
                eprintln!("cannot read {path}: {error}");
                return ExitCode::from(2);
            }
        }
    }

    let mut samples: Vec<Sample> = Vec::new();
    let mut validation_fens: Vec<(String, f32)> = Vec::new();
    let mut rng = Rng::new(settings.seed);
    let mut train: Vec<Sample> = Vec::new();
    let mut validation: Vec<Sample> = Vec::new();
    'outer: for text in &texts {
        for raw in text.lines() {
            if samples.len() + train.len() + validation.len() >= settings.limit {
                break 'outer;
            }
            let Some(line) = parse_line(raw) else {
                continue;
            };
            let Ok(pos) = Position::from_fen(line.fen) else {
                continue;
            };
            let sample = build_sample(&pos, target_of(&line, settings.lambda));
            if rng.unit() < settings.validation {
                if validation_fens.len() < 3000 {
                    validation_fens.push((line.fen.to_string(), sample.target));
                }
                validation.push(sample);
            } else {
                train.push(sample);
            }
        }
    }
    samples.clear();
    drop(texts);
    eprintln!(
        "loaded {} training and {} validation positions in {:.1}s",
        train.len(),
        validation.len(),
        started.elapsed().as_secs_f64()
    );
    if train.is_empty() {
        eprintln!("no training data");
        return ExitCode::from(2);
    }

    let mut params = match &settings.resume {
        Some(path) => match Params::load_raw(path) {
            Ok(p) => p,
            Err(message) => {
                eprintln!("{message}");
                return ExitCode::from(2);
            }
        },
        None => Params::random(&mut rng),
    };
    let mut adam = Adam::new();
    let raw_path = format!("{}.params", settings.out);

    let mut order: Vec<usize> = (0..train.len()).collect();
    for epoch in 0..settings.epochs {
        let progress = if settings.epochs > 1 {
            epoch as f32 / (settings.epochs - 1) as f32
        } else {
            0.0
        };
        let lr = settings.lr * (settings.lr_final / settings.lr).powf(progress);

        for i in (1..order.len()).rev() {
            let j = rng.below(i + 1);
            order.swap(i, j);
        }

        let mut epoch_loss = 0.0f64;
        let mut epoch_count = 0usize;
        for batch in order.chunks(settings.batch) {
            let chunk = batch.len().div_ceil(settings.threads);
            let mut total = Grad::zeros();
            let current = &params;
            let train_ref = &train;
            let grads: Vec<Grad> = std::thread::scope(|scope| {
                let handles: Vec<_> = batch
                    .chunks(chunk)
                    .map(|part| {
                        scope.spawn(move || {
                            let mut grad = Grad::zeros();
                            for &index in part {
                                backward(current, &train_ref[index], &mut grad);
                            }
                            grad
                        })
                    })
                    .collect();
                handles.into_iter().filter_map(|h| h.join().ok()).collect()
            });
            for grad in &grads {
                total.merge(grad);
            }
            epoch_loss += total.loss;
            epoch_count += total.count;
            let scale = 1.0 / total.count.max(1) as f32;
            adam.apply(&mut params, &total, lr, scale);
        }

        let val_loss = evaluate_set(&params, &validation, settings.threads);
        eprintln!(
            "epoch {:>3}/{} lr {:.6} train {:.6} validation {:.6} ({:.0}s)",
            epoch + 1,
            settings.epochs,
            lr,
            epoch_loss / epoch_count.max(1) as f64,
            val_loss,
            started.elapsed().as_secs_f64()
        );

        if let Err(error) = fs::write(&settings.out, params.to_bytes()) {
            eprintln!("cannot write {}: {error}", settings.out);
            return ExitCode::from(2);
        }
        if let Err(error) = params.save_raw(&raw_path) {
            eprintln!("cannot write {raw_path}: {error}");
        }
    }

    quantization_report(&params, &validation_fens);
    eprintln!("wrote {}", settings.out);
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_params(seed: u64) -> Params {
        let mut rng = Rng::new(seed);
        let mut p = Params::random(&mut rng);
        for w in p.w.iter_mut() {
            *w *= 3.0;
        }
        p.clamp();
        p
    }

    fn sample_for(fen: &str, target: f32) -> Sample {
        build_sample(&Position::from_fen(fen).expect("fen"), target)
    }

    #[test]
    fn analytic_gradient_matches_numeric_gradient() {
        let fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
        let sample = sample_for(fen, 0.62);
        let params = small_params(5);
        let mut grad = Grad::zeros();
        backward(&params, &sample, &mut grad);

        let loss = |p: &Params| {
            let e = sigmoid(forward(p, &sample)) - sample.target;
            (e * e) as f64
        };
        let eps = 1e-3f32;
        let mut checked = 0;
        let mut worst = 0.0f64;
        let touched: Vec<usize> = sample.us[..sample.count as usize]
            .iter()
            .map(|&f| f as usize * HIDDEN)
            .collect();
        for &base in touched.iter().take(4) {
            for h in (0..HIDDEN).step_by(11) {
                let mut plus = params.clone();
                let mut minus = params.clone();
                plus.w[base + h] += eps;
                minus.w[base + h] -= eps;
                let numeric = (loss(&plus) - loss(&minus)) / (2.0 * eps as f64);
                let analytic = grad.w[base + h] as f64;
                worst = worst.max((numeric - analytic).abs());
                checked += 1;
            }
        }
        for h in (0..2 * HIDDEN).step_by(17) {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus.ow[h] += eps;
            minus.ow[h] -= eps;
            let numeric = (loss(&plus) - loss(&minus)) / (2.0 * eps as f64);
            worst = worst.max((numeric - grad.ow[h] as f64).abs());
            checked += 1;
        }
        let mut plus = params.clone();
        let mut minus = params.clone();
        plus.ob += eps;
        minus.ob -= eps;
        let numeric = (loss(&plus) - loss(&minus)) / (2.0 * eps as f64);
        worst = worst.max((numeric - grad.ob as f64).abs());
        assert!(checked > 20);
        assert!(worst < 2e-3, "worst gradient difference {worst}");
    }

    #[test]
    fn exported_network_matches_the_float_model() {
        let mut params = small_params(9);
        for w in params.w.iter_mut().chain(params.b.iter_mut()) {
            *w = (*w * QA as f32).round() / QA as f32;
        }
        for w in params.ow.iter_mut() {
            *w = (*w * QB as f32).round() / QB as f32;
        }
        params.ob = (params.ob * (QA * QB) as f32).round() / (QA * QB) as f32;
        let net = nnue::Network::from_bytes(&params.to_bytes()).expect("export loads");
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
        ];
        for fen in fens {
            let pos = Position::from_fen(fen).expect("fen");
            let sample = build_sample(&pos, 0.5);
            let float_cp = forward(&params, &sample) * SCALE as f32;
            let mut acc = nnue::Accumulator::default();
            nnue::refresh(&net, &mut acc, &pos);
            let int_cp = nnue::evaluate(&net, &acc, pos.side_to_move()) as f32;
            assert!(
                (float_cp - int_cp).abs() < 2.0,
                "{fen}: float {float_cp} vs integer {int_cp}"
            );
        }
    }

    #[test]
    fn adam_reduces_loss_on_a_small_fixed_set() {
        let fens = [
            ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", 0.8),
            ("4k3/4p3/8/8/8/8/8/4K3 w - - 0 1", 0.2),
            ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 0.95),
            ("3qk3/8/8/8/8/8/8/4K3 w - - 0 1", 0.05),
        ];
        let samples: Vec<Sample> = fens.iter().map(|(f, t)| sample_for(f, *t)).collect();
        let mut rng = Rng::new(3);
        let mut params = Params::random(&mut rng);
        let mut adam = Adam::new();
        let before = evaluate_set(&params, &samples, 1);
        for _ in 0..200 {
            let mut grad = Grad::zeros();
            for s in &samples {
                backward(&params, s, &mut grad);
            }
            adam.apply(&mut params, &grad, 0.01, 1.0 / samples.len() as f32);
        }
        let after = evaluate_set(&params, &samples, 1);
        assert!(after < before * 0.2, "loss {before} -> {after}");
    }
}
