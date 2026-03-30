# fpo-rust

[![Crates.io](https://img.shields.io/crates/v/fpo-rust.svg)](https://crates.io/crates/fpo-rust)
[![Docs.rs](https://docs.rs/fpo-rust/badge.svg)](https://docs.rs/fpo-rust)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fast License Plate OCR inference in pure Rust. A high-performance port of [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) that provides fast and accurate license plate character recognition across multiple countries.

## Features

- **Pure Rust implementation** using `tract-onnx` for ONNX model inference
- **Multiple pre-trained models** for global and regional license plates
- **Character-level confidence scores** for result validation
- **Region detection** for plates with regional information (e.g., country/state)
- **Efficient batch processing** with automatic image resizing
- **Offline support** with model caching and custom model paths
- **Cross-platform** support (Linux, macOS, Windows)

## Installation

### As a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
fpo-rust = "0.1"
```

Or use the git repository:

```toml
[dependencies]
fpo-rust = { git = "https://github.com/alparslanahmed/fpo-rust" }
```

Check the [crates.io page](https://crates.io/crates/fpo-rust) for the latest version.

### As a CLI Tool

Clone the repository and build:

```bash
git clone https://github.com/alparslanahmed/fpo-rust
cd fpo-rust
cargo build --release
```

The binary will be available at `target/release/fpo-rust`.

## Quick Start

### Using as a Library

```rust
use fpo_rust::{LicensePlateRecognizer, OcrModel, PlateInput};

fn main() -> anyhow::Result<()> {
    // Load a model from the hub (downloads on first use, then cached)
    let recognizer = LicensePlateRecognizer::from_hub(
        OcrModel::CctSV2Global,  // Recommended global model
        false  // force_download
    )?;

    // Run inference on a single image
    let prediction = recognizer.run_one(
        PlateInput::from("path/to/plate.png"),
        true,   // return_confidence scores
        true    // remove_pad_char
    )?;

    println!("Plate: {}", prediction.plate);
    println!("Average Confidence: {:.2}", prediction.avg_char_confidence());
    if let Some(region) = &prediction.region {
        println!("Region: {} ({:.1}%)", region, prediction.region_prob.unwrap_or(0.0) * 100.0);
    }

    Ok(())
}
```

### Using the CLI

#### Run inference on images:

```bash
# Using a hub model (downloads model on first run)
fpo-rust run --model cct-s-v2-global-model plate1.jpg plate2.png

# Using a custom model and config
fpo-rust run --onnx ./models/custom.onnx --config ./models/custom_config.yaml plate.jpg

# Keep padding characters in output
fpo-rust run --model cct-s-v2-global-model --keep-pad plate.jpg
```

#### Run benchmark:

```bash
# Benchmark with 500 iterations, batch size 1
fpo-rust benchmark --model cct-s-v2-global-model

# Custom benchmark settings
fpo-rust benchmark --model cct-s-v2-global-model --iters 1000 --batch 32 --warmup 100

# Include pre/post-processing time in benchmark
fpo-rust benchmark --model cct-s-v2-global-model --include-processing
```

## Available Models

### Global Models (Recommended)

- **`cct-s-v2-global-model`** ⭐ - Compact Convolutional Transformer Small v2, optimized for global plates
- **`cct-xs-v2-global-model`** - XSmall v2, faster but less accurate
- **`cct-s-v1-global-model`** - Small v1, previous version
- **`cct-xs-v1-global-model`** - XSmall v1, lightweight

### Regional Models

- **`european-plates-mobile-vit-v2-model`** - Optimized for European license plates
- **`global-plates-mobile-vit-v2-model`** - Mobile-optimized global model
- **`argentinian-plates-cnn-model`** - CNN model for Argentinian plates
- **`argentinian-plates-cnn-synth-model`** - CNN model trained with synthetic data

## Offline Usage & Custom Models

### Automatic Caching

Models are automatically downloaded and cached on first use. The default cache location is:

- **Linux/macOS**: `~/.cache/fpo-rust/`
- **Windows**: `%APPDATA%\fpo-rust\`

### Using Custom Models

If you have your own ONNX model and config file:

```rust
use fpo_rust::{LicensePlateRecognizer, PlateInput};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let recognizer = LicensePlateRecognizer::from_files(
        Path::new("./models/my_model.onnx"),
        Path::new("./models/my_config.yaml")
    )?;

    let prediction = recognizer.run_one(
        PlateInput::from("plate.jpg"),
        true,
        true
    )?;

    println!("{}", prediction.plate);
    Ok(())
}
```

### Save Models to Custom Directory

Download models to a specific directory for offline use:

```rust
use fpo_rust::{LicensePlateRecognizer, OcrModel};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let custom_dir = Path::new("./offline_models");

    // Downloads to ./offline_models/ instead of default cache
    let recognizer = LicensePlateRecognizer::from_hub_to_dir(
        OcrModel::CctSV2Global,
        custom_dir,
        false
    )?;

    // Later, you can use the cached models directly
    let recognizer = LicensePlateRecognizer::from_files(
        custom_dir.join("cct_s_v2_global.onnx"),
        custom_dir.join("cct_s_v2_global_plate_config.yaml")
    )?;

    Ok(())
}
```

### CLI: Download Models Offline

Download all models to a directory for offline use:

```bash
# Create a models directory
mkdir models

# Run CLI with custom model path (this will download if not present)
fpo-rust run --onnx ./models/my_model.onnx --config ./models/my_config.yaml plate.jpg
```

## API Reference

### Main Types

#### `LicensePlateRecognizer`

The main inference engine.

**Methods:**

- `from_hub(model: OcrModel, force_download: bool) -> Result<Self>`
  - Load a model from the hub with automatic caching

- `from_hub_to_dir(model: OcrModel, save_dir: &Path, force_download: bool) -> Result<Self>`
  - Load a model from hub, saving to a specific directory

- `from_files(onnx_path: impl AsRef<Path>, config_path: impl AsRef<Path>) -> Result<Self>`
  - Load a custom ONNX model with its config

- `run(inputs: &[PlateInput], return_confidence: bool, remove_pad_char: bool) -> Result<Vec<PlatePrediction>>`
  - Run inference on multiple images

- `run_one(input: PlateInput, return_confidence: bool, remove_pad_char: bool) -> Result<PlatePrediction>`
  - Run inference on a single image

#### `PlateInput<'a>`

Represents a single plate image input. Can be:

```rust
// From file path
PlateInput::from("plate.jpg")

// From Path object
PlateInput::from(Path::new("plate.jpg"))

// From pre-loaded DynamicImage
use image::open;
let img = open("plate.jpg")?;
PlateInput::from(img)
```

#### `PlatePrediction`

Output of a single inference.

**Fields:**

- `plate: String` - Recognized plate text
- `region: Option<String>` - Region/country if detected
- `region_prob: Option<f32>` - Confidence of region (0.0-1.0)
- `char_probs: Option<Vec<f32>>` - Per-character confidence scores

**Methods:**

- `avg_char_confidence() -> f32` - Average confidence across all characters

#### `OcrModel`

Enum of available hub models:

```rust
pub enum OcrModel {
    CctSV2Global,
    CctXsV2Global,
    CctSV1Global,
    CctXsV1Global,
    // ... and more regional models
}
```

## Output Format

### Library Output

```rust
let pred = recognizer.run_one(input, true, true)?;
println!("Plate: {}", pred.plate);
println!("Avg Confidence: {:.2}", pred.avg_char_confidence());
println!("Region: {}", pred.region.unwrap_or("Unknown".to_string()));
```

### CLI Output

```
plate.jpg: 34PE7523 [Turkey] (98.7%) - Char Confidence: 0.99
```

- `34PE7523` - Recognized plate text
- `[Turkey]` - Detected region (if available)
- `(98.7%)` - Region confidence score
- `0.99` - Average character confidence (0.0-1.0)

## Performance

Performance varies by model size and hardware. Example benchmarks on standard hardware:

| Model | Inference Time | Memory |
|-------|----------------|--------|
| cct-xs-v2-global | ~10-15ms | ~50MB |
| cct-s-v2-global | ~20-30ms | ~100MB |
| mobile-vit-v2-global | ~30-50ms | ~150MB |

Use `fpo-rust benchmark --model <name>` to measure performance on your hardware.

## Configuration

The library respects the `XDG_CACHE_HOME` environment variable on Linux/macOS. To use a custom cache directory:

```bash
# Use custom cache directory
export XDG_CACHE_HOME=/path/to/cache
cargo run -- run --model cct-s-v2-global-model plate.jpg
```

## Building for Production

### Optimize the binary:

```bash
cargo build --release
```

The optimized binary will be at `target/release/fpo-rust`.

### Static linking (optional):

For maximum portability, you can create a statically-linked binary. This requires additional setup depending on your platform.

## Troubleshooting

### Model Download Fails

- Check your internet connection
- Verify GitHub is not blocked
- Use `RUST_LOG=debug` for more details:
  ```bash
  RUST_LOG=debug cargo run -- run --model cct-s-v2-global-model plate.jpg
  ```

### Low Confidence Scores

- Try a different model optimized for your region
- Ensure plate images have sufficient contrast and resolution
- Pre-process images (crop, rotate) if plates are at odd angles

### Out of Memory

- Use a smaller model (e.g., `cct-xs-v2-global-model`)
- Reduce batch size in benchmarks: `--batch 1`

## Development

### Running Tests

```bash
cargo test
```

### Running Benchmarks

```bash
cargo run --release -- benchmark --model cct-s-v2-global-model --iters 1000
```

## Dependencies

- **tract-onnx** - Pure-Rust ONNX runtime
- **image** - Image loading and processing
- **serde** - Serialization framework
- **serde_yml** - YAML parsing for config files
- **ureq** - Lightweight HTTP client for model downloads
- **anyhow** - Error handling

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) - Original Python implementation
- [tract](https://github.com/snipsco/tract) - Pure Rust ONNX runtime
- [ONNX](https://onnx.ai/) - Open Neural Network Exchange format

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
