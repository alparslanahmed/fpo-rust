//! fpo-rust – Rust inference port of [fast-plate-ocr].
//!
//! [fast-plate-ocr]: https://github.com/ankandrew/fast-plate-ocr
//!
//! # Quick start
//!
//! ```no_run
//! use fpo_rust::{LicensePlateRecognizer, PlateInput, OcrModel};
//!
//! let rec = LicensePlateRecognizer::from_hub(OcrModel::CctSV2Global, false).unwrap();
//! let pred = rec.run_one(PlateInput::from("plate.png"), false, true).unwrap();
//! println!("{}", pred.plate);
//! ```

pub mod config;
pub mod hub;
pub mod process;
pub mod recognizer;

pub use config::PlateConfig;
pub use hub::OcrModel;
pub use process::PlatePrediction;
pub use recognizer::{LicensePlateRecognizer, PlateInput};
