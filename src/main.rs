//! fpo-rust CLI - run fast-plate-ocr inference from the command line.

use anyhow::{bail, Context};
use fpo_rust::{hub::download_model, LicensePlateRecognizer, OcrModel, PlateConfig, PlateInput};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    process::{self, Command},
};

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
    run           Run OCR inference on plate image(s)
    benchmark     Run throughput benchmark
    convert-ncnn  Convert an ONNX model to Tencent NCNN .param/.bin files with pnnx

COMMON MODEL OPTIONS:
    --backend <onnx|ncnn>  Inference backend (default: onnx)
    --model <NAME>         Hub model name (e.g. cct-s-v2-global-model)
    --onnx <PATH>          Path to a custom ONNX model file
    --config <PATH>        Path to the matching plate config YAML

NCNN MODEL OPTIONS:
    --param <PATH>         Path to converted NCNN .param file
    --bin <PATH>           Path to converted NCNN .bin file
    --input-name <NAME>    NCNN input blob name (default: inferred, then in0)
    --plate-output <NAME>  NCNN plate output blob name (default: inferred, then out0)
    --region-output <NAME> NCNN region output blob name (default: inferred, then out1)
    --no-region-output     Disable NCNN region extraction
    --threads <N>          NCNN worker threads
    --vulkan               Enable NCNN Vulkan compute if the binary and NCNN were built for it

OPTIONS (run):
    --keep-pad             Keep trailing padding characters in output
    IMAGES...              One or more image paths to process (confidence scores always shown)

OPTIONS (benchmark):
    --iters <N>            Number of timed iterations (default: 500)
    --batch <N>            Batch size (default: 1)
    --warmup <N>           Warm-up iterations (default: 50)
    --include-processing   Include pre/post-processing in timing

OPTIONS (convert-ncnn):
    --model <NAME>         Download and convert a hub model
    --onnx <PATH>          Convert a custom ONNX model
    --config <PATH>        Plate config used to set pnnx inputshape for custom ONNX
    --out-dir <PATH>       Output directory (default: ONNX directory or hub cache dir)
    --pnnx <PATH>          pnnx executable (default: pnnx)
    --inputshape <SHAPE>   Override pnnx inputshape, e.g. [1,70,140,1]u8
    --fp16                 Let pnnx store fp16 weights (default: fp16=0 for portability)
    --force                Re-run conversion even if output files already exist

AVAILABLE HUB MODELS:
    cct-s-v2-global-model (recommended)
    cct-xs-v2-global-model
    cct-s-v1-global-model
    cct-xs-v1-global-model
    cct-s-relu-v1-global-model
    cct-xs-relu-v1-global-model
    argentinian-plates-cnn-model
    argentinian-plates-cnn-synth-model
    european-plates-mobile-vit-v2-model
    global-plates-mobile-vit-v2-model
",
        version = env!("CARGO_PKG_VERSION")
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Onnx,
    Ncnn,
}

impl BackendChoice {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "onnx" | "tract" => Ok(Self::Onnx),
            "ncnn" => Ok(Self::Ncnn),
            _ => bail!("Unknown backend '{value}'. Use 'onnx' or 'ncnn'."),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct NcnnCliOptions {
    param: Option<PathBuf>,
    bin: Option<PathBuf>,
    input_name: Option<String>,
    plate_output_name: Option<String>,
    region_output_name: Option<String>,
    disable_region_output: bool,
    num_threads: Option<u32>,
    use_vulkan: bool,
}

impl NcnnCliOptions {
    fn any_model_option_set(&self) -> bool {
        self.param.is_some()
            || self.bin.is_some()
            || self.input_name.is_some()
            || self.plate_output_name.is_some()
            || self.region_output_name.is_some()
            || self.disable_region_output
            || self.num_threads.is_some()
            || self.use_vulkan
    }
}

fn parse_common_model_arg(
    arg: &str,
    iter: &mut std::slice::Iter<'_, String>,
    backend: &mut BackendChoice,
    model: &mut Option<String>,
    onnx: &mut Option<PathBuf>,
    cfg: &mut Option<PathBuf>,
    ncnn: &mut NcnnCliOptions,
) -> anyhow::Result<bool> {
    match arg {
        "--backend" => {
            *backend = BackendChoice::parse(iter.next().context("--backend requires a value")?)?;
            Ok(true)
        }
        "--model" => {
            *model = Some(iter.next().context("--model requires a value")?.clone());
            Ok(true)
        }
        "--onnx" => {
            *onnx = Some(
                iter.next()
                    .context("--onnx requires a value")?
                    .clone()
                    .into(),
            );
            Ok(true)
        }
        "--config" => {
            *cfg = Some(
                iter.next()
                    .context("--config requires a value")?
                    .clone()
                    .into(),
            );
            Ok(true)
        }
        "--param" => {
            ncnn.param = Some(
                iter.next()
                    .context("--param requires a value")?
                    .clone()
                    .into(),
            );
            Ok(true)
        }
        "--bin" => {
            ncnn.bin = Some(
                iter.next()
                    .context("--bin requires a value")?
                    .clone()
                    .into(),
            );
            Ok(true)
        }
        "--input-name" => {
            ncnn.input_name = Some(
                iter.next()
                    .context("--input-name requires a value")?
                    .clone(),
            );
            Ok(true)
        }
        "--plate-output" => {
            ncnn.plate_output_name = Some(
                iter.next()
                    .context("--plate-output requires a value")?
                    .clone(),
            );
            Ok(true)
        }
        "--region-output" => {
            ncnn.region_output_name = Some(
                iter.next()
                    .context("--region-output requires a value")?
                    .clone(),
            );
            Ok(true)
        }
        "--no-region-output" => {
            ncnn.disable_region_output = true;
            Ok(true)
        }
        "--threads" => {
            ncnn.num_threads = Some(
                iter.next()
                    .context("--threads requires a value")?
                    .parse()
                    .context("--threads must be an integer")?,
            );
            Ok(true)
        }
        "--vulkan" => {
            ncnn.use_vulkan = true;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn build_recognizer(
    backend: BackendChoice,
    model: Option<&str>,
    onnx: Option<&PathBuf>,
    cfg: Option<&PathBuf>,
    ncnn: &NcnnCliOptions,
) -> anyhow::Result<LicensePlateRecognizer> {
    match backend {
        BackendChoice::Onnx => {
            if ncnn.any_model_option_set() {
                bail!("NCNN options require --backend ncnn");
            }
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
        BackendChoice::Ncnn => build_ncnn_recognizer(model, onnx, cfg, ncnn),
    }
}

#[cfg(feature = "ncnn")]
fn build_ncnn_recognizer(
    model: Option<&str>,
    onnx: Option<&PathBuf>,
    cfg: Option<&PathBuf>,
    ncnn: &NcnnCliOptions,
) -> anyhow::Result<LicensePlateRecognizer> {
    let options = fpo_rust::NcnnOptions {
        input_name: ncnn.input_name.clone(),
        plate_output_name: ncnn.plate_output_name.clone(),
        region_output_name: ncnn.region_output_name.clone(),
        disable_region_output: ncnn.disable_region_output,
        num_threads: ncnn.num_threads,
        use_vulkan: ncnn.use_vulkan,
    };

    match (model, onnx, cfg, ncnn.param.as_ref(), ncnn.bin.as_ref()) {
        (Some(name), None, None, None, None) => {
            let ocr_model =
                OcrModel::from_str(name).with_context(|| format!("Unknown hub model: '{name}'"))?;
            LicensePlateRecognizer::from_hub_ncnn_with_options(ocr_model, false, options)
        }
        (None, Some(onnx_path), Some(cfg_path), None, None) => {
            let (param_path, bin_path) = ncnn_paths_for_onnx(onnx_path, None)?;
            if !param_path.is_file() || !bin_path.is_file() {
                bail!(
                    "Converted NCNN files are missing. Run: fpo-rust convert-ncnn --onnx {} --config {}",
                    onnx_path.display(),
                    cfg_path.display()
                );
            }
            LicensePlateRecognizer::from_ncnn_files_with_options(
                param_path, bin_path, cfg_path, options,
            )
        }
        (None, None, Some(cfg_path), Some(param_path), Some(bin_path)) => {
            LicensePlateRecognizer::from_ncnn_files_with_options(
                param_path, bin_path, cfg_path, options,
            )
        }
        _ => bail!(
            "For --backend ncnn, specify --model <NAME>, or --onnx <PATH> --config <PATH>, or --param <PATH> --bin <PATH> --config <PATH>."
        ),
    }
}

#[cfg(not(feature = "ncnn"))]
fn build_ncnn_recognizer(
    _model: Option<&str>,
    _onnx: Option<&PathBuf>,
    _cfg: Option<&PathBuf>,
    _ncnn: &NcnnCliOptions,
) -> anyhow::Result<LicensePlateRecognizer> {
    bail!(
        "NCNN backend requested, but this binary was built without NCNN support. Rebuild with: cargo build --release --features ncnn-cpu"
    );
}

fn cmd_run(args: &[String]) -> anyhow::Result<()> {
    let mut backend = BackendChoice::Onnx;
    let mut model: Option<String> = None;
    let mut onnx: Option<PathBuf> = None;
    let mut cfg: Option<PathBuf> = None;
    let mut ncnn = NcnnCliOptions::default();
    let mut keep_pad = false;
    let mut images: Vec<String> = vec![];

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if parse_common_model_arg(
            arg.as_str(),
            &mut iter,
            &mut backend,
            &mut model,
            &mut onnx,
            &mut cfg,
            &mut ncnn,
        )? {
            continue;
        }

        match arg.as_str() {
            "--keep-pad" => keep_pad = true,
            other => images.push(other.to_owned()),
        }
    }

    if images.is_empty() {
        bail!("No images specified. Pass one or more image paths.");
    }

    let rec = build_recognizer(
        backend,
        model.as_deref(),
        onnx.as_ref(),
        cfg.as_ref(),
        &ncnn,
    )?;

    let inputs: Vec<PlateInput<'_>> = images
        .iter()
        .map(|s| PlateInput::from(s.as_str()))
        .collect();

    let predictions = rec.run(&inputs, true, !keep_pad)?;

    for (path, pred) in images.iter().zip(predictions.iter()) {
        print!("{path}: {}", pred.plate);
        if let Some(region) = &pred.region {
            print!(" [{region}]");
            if let Some(rp) = pred.region_prob {
                print!(" ({:.1}%)", rp * 100.0);
            }
        }
        if let Some(probs) = &pred.char_probs {
            let avg_conf: f32 = probs.iter().sum::<f32>() / probs.len() as f32;
            print!(" - Char Confidence: {:.2}", avg_conf);
        }
        println!();
    }

    Ok(())
}

fn cmd_benchmark(args: &[String]) -> anyhow::Result<()> {
    let mut backend = BackendChoice::Onnx;
    let mut model: Option<String> = None;
    let mut onnx: Option<PathBuf> = None;
    let mut cfg: Option<PathBuf> = None;
    let mut ncnn = NcnnCliOptions::default();
    let mut n_iter: usize = 500;
    let mut batch_size: usize = 1;
    let mut warmup: usize = 50;
    let mut include_processing = false;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if parse_common_model_arg(
            arg.as_str(),
            &mut iter,
            &mut backend,
            &mut model,
            &mut onnx,
            &mut cfg,
            &mut ncnn,
        )? {
            continue;
        }

        match arg.as_str() {
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

    let rec = build_recognizer(
        backend,
        model.as_deref(),
        onnx.as_ref(),
        cfg.as_ref(),
        &ncnn,
    )?;
    rec.benchmark(n_iter, batch_size, warmup, include_processing)?;
    Ok(())
}

fn cmd_convert_ncnn(args: &[String]) -> anyhow::Result<()> {
    let mut model: Option<String> = None;
    let mut onnx: Option<PathBuf> = None;
    let mut cfg: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut pnnx = PathBuf::from("pnnx");
    let mut force = false;
    let mut fp16 = false;
    let mut inputshape: Option<String> = None;

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
            "--out-dir" => {
                out_dir = Some(
                    iter.next()
                        .context("--out-dir requires a value")?
                        .clone()
                        .into(),
                );
            }
            "--pnnx" => {
                pnnx = iter
                    .next()
                    .context("--pnnx requires a value")?
                    .clone()
                    .into();
            }
            "--inputshape" => {
                inputshape = Some(
                    iter.next()
                        .context("--inputshape requires a value")?
                        .clone(),
                );
            }
            "--fp16" => fp16 = true,
            "--force" => force = true,
            other => bail!("Unknown flag: {other}"),
        }
    }

    let (onnx_path, cfg_path) = match (model.as_deref(), onnx.as_ref()) {
        (Some(_), Some(_)) => bail!("Specify either --model or --onnx, not both."),
        (Some(name), None) => {
            let ocr_model =
                OcrModel::from_str(name).with_context(|| format!("Unknown hub model: '{name}'"))?;
            download_model(&ocr_model, out_dir.as_deref(), false)?
        }
        (None, Some(onnx_path)) => (onnx_path.clone(), cfg.clone().unwrap_or_default()),
        (None, None) => bail!("Specify --model <NAME> or --onnx <PATH>."),
    };

    if !onnx_path.is_file() {
        bail!("ONNX model not found: {}", onnx_path.display());
    }

    let output_dir = out_dir
        .clone()
        .or_else(|| onnx_path.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("Cannot create output dir {}", output_dir.display()))?;

    let (param_path, bin_path) = ncnn_paths_for_onnx(&onnx_path, Some(&output_dir))?;
    if !force && param_path.is_file() && bin_path.is_file() {
        println!("NCNN files already exist:");
        println!("  {}", param_path.display());
        println!("  {}", bin_path.display());
        return Ok(());
    }

    let inputshape = match inputshape {
        Some(shape) => Some(shape),
        None if cfg_path.is_file() => {
            let cfg = PlateConfig::from_yaml(&cfg_path)?;
            Some(format!(
                "[1,{},{},{}]u8",
                cfg.img_height,
                cfg.img_width,
                cfg.num_channels()
            ))
        }
        None => None,
    };

    let mut command = Command::new(&pnnx);
    command
        .arg(&onnx_path)
        .arg(format!("ncnnparam={}", param_path.display()))
        .arg(format!("ncnnbin={}", bin_path.display()))
        .arg(format!("fp16={}", if fp16 { 1 } else { 0 }));

    if let Some(shape) = inputshape {
        command.arg(format!("inputshape={shape}"));
    }

    let status = command
        .status()
        .with_context(|| format!("Failed to start pnnx at {}", pnnx.display()))?;
    if !status.success() {
        bail!("pnnx conversion failed with status {status}");
    }

    let removed_layers = scrub_unsupported_ncnn_layers(&param_path)?;

    println!("Wrote NCNN model:");
    println!("  {}", param_path.display());
    println!("  {}", bin_path.display());
    if removed_layers > 0 {
        println!("Removed unsupported NCNN no-op layers: {removed_layers}");
    }
    if cfg_path.is_file() {
        println!("Use with config:");
        println!("  {}", cfg_path.display());
    }
    Ok(())
}

fn scrub_unsupported_ncnn_layers(param_path: &Path) -> anyhow::Result<usize> {
    let text = std::fs::read_to_string(param_path)
        .with_context(|| format!("Cannot read NCNN param: {}", param_path.display()))?;
    let lines: Vec<String> = text.lines().map(str::to_owned).collect();
    if lines.len() < 2 {
        bail!("Invalid NCNN param file: {}", param_path.display());
    }

    let header = lines[1].split_whitespace().collect::<Vec<_>>();
    if header.len() != 2 {
        bail!(
            "Invalid NCNN param header in {}: {}",
            param_path.display(),
            lines[1]
        );
    }

    let mut aliases = HashMap::<String, String>::new();
    let mut removed = HashSet::<usize>::new();

    for (index, line) in lines.iter().enumerate().skip(2) {
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first() == Some(&"Tensor.to") {
            if tokens.len() >= 6 && tokens[2] == "1" && tokens[3] == "1" {
                aliases.insert(tokens[5].to_owned(), tokens[4].to_owned());
                removed.insert(index);
            } else {
                bail!("Unsupported Tensor.to shape in {}", param_path.display());
            }
        }
    }

    if aliases.is_empty() {
        return Ok(0);
    }

    let mut output = vec![lines[0].clone(), String::new()];
    let mut top_names = HashSet::<String>::new();

    for (index, line) in lines.iter().enumerate().skip(2) {
        if removed.contains(&index) {
            continue;
        }

        let mut tokens = line
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        if tokens.len() >= 4 {
            let bottom_count = tokens[2].parse::<usize>().with_context(|| {
                format!("Invalid NCNN bottom count in {}", param_path.display())
            })?;
            let top_count = tokens[3]
                .parse::<usize>()
                .with_context(|| format!("Invalid NCNN top count in {}", param_path.display()))?;

            if tokens.len() < 4 + bottom_count + top_count {
                bail!("Invalid NCNN layer line in {}", param_path.display());
            }

            for token in tokens.iter_mut().skip(4).take(bottom_count) {
                *token = resolve_ncnn_alias(&aliases, token);
            }

            for token in tokens.iter().skip(4 + bottom_count).take(top_count) {
                top_names.insert(token.clone());
            }
        }

        output.push(tokens.join(" "));
    }

    output[1] = format!("{} {}", output.len() - 2, top_names.len());
    std::fs::write(param_path, format!("{}\n", output.join("\n")))
        .with_context(|| format!("Cannot write NCNN param: {}", param_path.display()))?;

    Ok(removed.len())
}

fn resolve_ncnn_alias(aliases: &HashMap<String, String>, name: &str) -> String {
    let mut current = name;
    let mut seen = HashSet::<&str>::new();

    while let Some(next) = aliases.get(current) {
        if !seen.insert(current) {
            break;
        }
        current = next;
    }

    current.to_owned()
}

fn ncnn_paths_for_onnx(
    onnx_path: &Path,
    out_dir: Option<&Path>,
) -> anyhow::Result<(PathBuf, PathBuf)> {
    let parent = out_dir
        .map(Path::to_path_buf)
        .or_else(|| onnx_path.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = onnx_path
        .file_stem()
        .with_context(|| format!("ONNX path has no file stem: {}", onnx_path.display()))?
        .to_string_lossy();
    Ok((
        parent.join(format!("{stem}.ncnn.param")),
        parent.join(format!("{stem}.ncnn.bin")),
    ))
}

#[cfg(test)]
mod tests {
    use super::scrub_unsupported_ncnn_layers;
    use std::io::Write;

    #[test]
    fn scrubs_tensor_to_noop_layer() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write!(
            file,
            "{}",
            "\
7767517
3 3
Input in0 0 1 in0
Tensor.to Tensor.to_0 1 1 in0 1
BinaryOp mul_0 1 1 1 out 0=2 1=1 2=3.92156886e-3
"
        )
        .unwrap();

        let removed = scrub_unsupported_ncnn_layers(file.path()).unwrap();
        assert_eq!(removed, 1);

        let text = std::fs::read_to_string(file.path()).unwrap();
        assert_eq!(
            text,
            "\
7767517
2 2
Input in0 0 1 in0
BinaryOp mul_0 1 1 in0 out 0=2 1=1 2=3.92156886e-3
"
        );
    }
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
        "convert-ncnn" => cmd_convert_ncnn(&args[2..]),
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
