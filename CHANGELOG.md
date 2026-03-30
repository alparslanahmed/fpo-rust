# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-03-30

### Added

- Initial release of fpo-rust
- Pure Rust ONNX-based license plate OCR inference using tract-onnx
- Support for multiple pre-trained models (global and regional)
- Character-level confidence scores for result validation
- Region detection for plates with regional information
- Efficient batch processing with automatic image resizing
- Offline support with model caching
- Custom model and config file support
- CLI tool with `run` and `benchmark` subcommands
- Comprehensive documentation and examples
- Average character confidence calculation
- Support for Linux, macOS, and Windows

### Features

- `LicensePlateRecognizer::from_hub()` - Load models from hub with caching
- `LicensePlateRecognizer::from_hub_to_dir()` - Download models to custom directory
- `LicensePlateRecognizer::from_files()` - Load custom ONNX models
- `run()` - Batch inference on multiple images
- `run_one()` - Single image inference
- CLI benchmarking with customizable parameters
- Full API documentation

[0.1.0]: https://github.com/alparslanahmed/fpo-rust/releases/tag/v0.1.0

