//! fpo-rust CLI – run fast-plate-ocr inference from the command line.
//!
//! Usage examples:
//!
//! ```text
//! # Run a hub model on one or more plate images
//! fpo-rust run --model cct-s-v2-global-model plate1.jpg plate2.png
//!
//! # Run with custom ONNX model + config
//! fpo-rust run --onnx my_model.onnx --config my_config.yaml plate.jpg
//!
//! # Benchmark
//! fpo-rust benchmark --model cct-s-v2-global-model
//! ```

use anyhow::{bail, Context};
use fpo_rust::{LicensePlateRecognizer, OcrModel, PlateInput};
use std::{path::PathBuf, process};

// ---------------------------------------------------------------------------
// Minimal arg-parsing (no extra dependency)
// ---------------------------------------------------------------------------

fn print_help() {
    println!(
        "fpo-rust {version}
Rust inference for fast-plate-ocr (https://github.com/ankandrew/fast-plate-ocr)

USAGE:
    fpo-rust <SUBCOMMAND> [OPTIONS] [IMAGES...]

SUBCOMMANDS:
    run         Run OCR inference on plate image(s)
    benchmark   Run throughput benchmark

OPTIONS (run):
    --model <NAME>       Hub model name (e.g. cct-s-v2-global-model)
    --onnx  <PATH>       Path to a custom ONNX model file
    --config <PATH>      Path to the matching plate config YAML
    --confidence         Print per-character confidence scores
    --keep-pad           Keep trailing padding characters in output
    IMAGES...            One or more image paths to process

OPTIONS (benchmark):
    --model <NAME>       Hub model name
    --onnx  <PATH>       Custom ONNX model
    --config <PATH>      Custom plate config
    --iters <N>          Number of timed iterations (default: 500)
    --batch <N>          Batch size (default: 1)
    --warmup <N>         Warm-up iterations (default: 50)
    --include-processing Include pre/post-processing in timing

AVAILABLE HUB MODELS:
    cct-s-v2-global-model (recommended)
    cct-xs-v2-global-model
    cct-s-v1-global-model
    cct-xs-v1-global-model
    argentinian-plates-cnn-model
    argentinian-plates-cnn-synth-model
    european-plates-mobile-vit-v2-model
    global-plates-mobile-vit-v2-model
",
        version = env!("CARGO_PKG_VERSION")
    );
}

fn build_recognizer(
    model: Option<&str>,
    onnx: Option<&PathBuf>,
    cfg: Option<&PathBuf>,
) -> anyhow::Result<LicensePlateRecognizer> {
    match (model, onnx, cfg) {
        (Some(name), None, None) => {
            let ocr_model = OcrModel::from_str(name)
                .with_context(|| format!("Unknown hub model: '{name}'"))?;
            LicensePlateRecognizer::from_hub(ocr_model, false)
        }
        (None, Some(onnx_path), Some(cfg_path)) => {
            LicensePlateRecognizer::from_files(onnx_path, cfg_path)
        }
        _ => bail!(
            "Specify either --model <NAME> or both --onnx <PATH> and --config <PATH>."
        ),
    }
}

fn cmd_run(args: &[String]) -> anyhow::Result<()> {
    let mut model: Option<String> = None;
    let mut onnx: Option<PathBuf> = None;
    let mut cfg: Option<PathBuf> = None;
    let mut confidence = false;
    let mut keep_pad = false;
    let mut images: Vec<String> = vec![];

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" => {
                model = Some(iter.next().context("--model requires a value")?.clone());
            }
            "--onnx" => {
                onnx = Some(
                    iter.next()
                        .context("--onnx requires a value")?
                        .clone()
                        .into(),
                );
            }
            "--config" => {
                cfg = Some(
                    iter.next()
                        .context("--config requires a value")?
                        .clone()
                        .into(),
                );
            }
            "--confidence" => confidence = true,
            "--keep-pad" => keep_pad = true,
            other => images.push(other.to_owned()),
        }
    }

    if images.is_empty() {
        bail!("No images specified. Pass one or more image paths.");
    }

    let rec = build_recognizer(model.as_deref(), onnx.as_ref(), cfg.as_ref())?;

    let inputs: Vec<PlateInput<'_>> = images
        .iter()
        .map(|s| PlateInput::from(s.as_str()))
        .collect();

    let predictions = rec.run(&inputs, confidence, !keep_pad)?;

    for (path, pred) in images.iter().zip(predictions.iter()) {
        print!("{path}: {}", pred.plate);
        if let Some(region) = &pred.region {
            print!(" [{region}]");
            if let Some(rp) = pred.region_prob {
                print!(" ({:.1}%)", rp * 100.0);
            }
        }
        println!();
        if let Some(probs) = &pred.char_probs {
            let chars: Vec<String> = pred
                .plate
                .chars()
                .zip(probs.iter())
                .map(|(c, p)| format!("{c}:{:.2}", p))
                .collect();
            println!("  confidences: [{}]", chars.join(", "));
        }
    }

    Ok(())
}

fn cmd_benchmark(args: &[String]) -> anyhow::Result<()> {
    let mut model: Option<String> = None;
    let mut onnx: Option<PathBuf> = None;
    let mut cfg: Option<PathBuf> = None;
    let mut n_iter: usize = 500;
    let mut batch_size: usize = 1;
    let mut warmup: usize = 50;
    let mut include_processing = false;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" => model = Some(iter.next().context("--model requires a value")?.clone()),
            "--onnx" => {
                onnx = Some(
                    iter.next()
                        .context("--onnx requires a value")?
                        .clone()
                        .into(),
                )
            }
            "--config" => {
                cfg = Some(
                    iter.next()
                        .context("--config requires a value")?
                        .clone()
                        .into(),
                )
            }
            "--iters" => {
                n_iter = iter
                    .next()
                    .context("--iters requires a value")?
                    .parse()
                    .context("--iters must be an integer")?;
            }
            "--batch" => {
                batch_size = iter
                    .next()
                    .context("--batch requires a value")?
                    .parse()
                    .context("--batch must be an integer")?;
            }
            "--warmup" => {
                warmup = iter
                    .next()
                    .context("--warmup requires a value")?
                    .parse()
                    .context("--warmup must be an integer")?;
            }
            "--include-processing" => include_processing = true,
            other => bail!("Unknown flag: {other}"),
        }
    }

    let rec = build_recognizer(model.as_deref(), onnx.as_ref(), cfg.as_ref())?;
    rec.benchmark(n_iter, batch_size, warmup, include_processing)?;
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_help();
        process::exit(0);
    }

    let result = match args[1].as_str() {
        "run" => cmd_run(&args[2..]),
        "benchmark" => cmd_benchmark(&args[2..]),
        "--help" | "-h" | "help" => {
            print_help();
            process::exit(0);
        }
        other => {
            eprintln!("Unknown subcommand: '{other}'. Run with --help for usage.");
            process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {e:#}");
        process::exit(1);
    }
}
