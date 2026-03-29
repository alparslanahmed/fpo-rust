//! Integration tests for fpo-rust.
//!
//! These tests do NOT require a real ONNX model or network access – they only test the
//! pure-Rust pre-/post-processing logic and config parsing.

use fpo_rust::{
    config::{ImageColorMode, ImageInterpolation, PaddingColor, PlateConfig},
    process::{postprocess_output, resize_image},
};
use image::{DynamicImage, ImageBuffer, Luma};
use std::io::Write;
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// PlateConfig YAML parsing
// ---------------------------------------------------------------------------

fn write_yaml(yaml: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(yaml.as_bytes()).unwrap();
    f
}

#[test]
fn test_plate_config_basic_parse() {
    let yaml = r#"
max_plate_slots: 7
alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
pad_char: "_"
img_height: 70
img_width: 140
"#;
    let f = write_yaml(yaml);
    let cfg = PlateConfig::from_yaml(f.path()).unwrap();

    assert_eq!(cfg.max_plate_slots, 7);
    assert_eq!(cfg.pad_char, '_');
    assert_eq!(cfg.img_height, 70);
    assert_eq!(cfg.img_width, 140);
    assert_eq!(cfg.num_channels(), 1); // default: grayscale
    assert!(!cfg.has_region_recognition());
}

#[test]
fn test_plate_config_rgb_with_regions() {
    let yaml = r#"
max_plate_slots: 8
alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
pad_char: "_"
img_height: 64
img_width: 256
image_color_mode: "rgb"
plate_regions:
  - "Argentina"
  - "Brazil"
  - "Chile"
"#;
    let f = write_yaml(yaml);
    let cfg = PlateConfig::from_yaml(f.path()).unwrap();

    assert_eq!(cfg.num_channels(), 3);
    assert!(cfg.has_region_recognition());
    assert_eq!(cfg.plate_regions.as_ref().unwrap().len(), 3);
}

#[test]
fn test_plate_config_defaults() {
    let yaml = r#"
max_plate_slots: 6
alphabet: "ABC_"
pad_char: "_"
img_height: 50
img_width: 100
"#;
    let f = write_yaml(yaml);
    let cfg = PlateConfig::from_yaml(f.path()).unwrap();

    // defaults
    assert_eq!(cfg.interpolation, ImageInterpolation::Linear);
    assert_eq!(cfg.image_color_mode, ImageColorMode::Grayscale);
    assert!(!cfg.keep_aspect_ratio);
}

#[test]
fn test_pad_idx() {
    let yaml = r#"
max_plate_slots: 4
alphabet: "0ABC_"
pad_char: "_"
img_height: 32
img_width: 64
"#;
    let f = write_yaml(yaml);
    let cfg = PlateConfig::from_yaml(f.path()).unwrap();
    assert_eq!(cfg.pad_idx(), 4); // '_' is index 4 in "0ABC_"
}

// ---------------------------------------------------------------------------
// Image resizing
// ---------------------------------------------------------------------------

fn gray_image(w: u32, h: u32, fill: u8) -> DynamicImage {
    DynamicImage::ImageLuma8(ImageBuffer::<Luma<u8>, _>::from_pixel(w, h, Luma([fill])))
}

#[test]
fn test_resize_exact_dimensions() {
    let img = gray_image(200, 100, 128);
    let resized = resize_image(
        img,
        70,
        140,
        &ImageColorMode::Grayscale,
        false,
        &ImageInterpolation::Linear,
        &PaddingColor::Gray(114),
    )
    .unwrap();
    assert_eq!(resized.width(), 140);
    assert_eq!(resized.height(), 70);
}

#[test]
fn test_resize_keep_aspect_ratio() {
    let img = gray_image(300, 100, 200);
    let resized = resize_image(
        img,
        70,
        140,
        &ImageColorMode::Grayscale,
        true,
        &ImageInterpolation::Linear,
        &PaddingColor::Gray(114),
    )
    .unwrap();
    // Output must always match the target dimensions.
    assert_eq!(resized.width(), 140);
    assert_eq!(resized.height(), 70);
}

// ---------------------------------------------------------------------------
// Post-processing
// ---------------------------------------------------------------------------

/// Build a fake model output where each slot is a one-hot vector
/// with the given index having the highest value.
fn fake_logits(slots: usize, vocab: usize, indices: &[usize]) -> Vec<f32> {
    assert_eq!(slots, indices.len());
    let mut out = vec![0.0f32; slots * vocab];
    for (s, &idx) in indices.iter().enumerate() {
        out[s * vocab + idx] = 1.0;
    }
    out
}

#[test]
fn test_postprocess_basic() {
    let alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
    let max_slots = 7;
    let vocab = alphabet.len();

    // Indices for "ABC1234" in the alphabet
    let abc = [10usize, 11, 12, 1, 2, 3, 4]; // A=10, B=11, C=12, 1=1, 2=2, 3=3, 4=4
    let logits = fake_logits(max_slots, vocab, &abc);

    let preds = postprocess_output(
        &logits,
        1,
        max_slots,
        alphabet,
        '_',
        true,
        false,
        None,
        None,
    )
    .unwrap();

    assert_eq!(preds.len(), 1);
    assert_eq!(preds[0].plate, "ABC1234");
    assert!(preds[0].char_probs.is_none());
}

#[test]
fn test_postprocess_strips_pad_char() {
    let alphabet = "ABC_";
    let max_slots = 6;
    let vocab = 4;
    // "AB__" → strip trailing '_' → "AB"
    let indices = [0usize, 1, 3, 3, 3, 3]; // A B _ _ _ _
    let logits = fake_logits(max_slots, vocab, &indices);

    let preds = postprocess_output(
        &logits,
        1,
        max_slots,
        alphabet,
        '_',
        true,
        false,
        None,
        None,
    )
    .unwrap();

    assert_eq!(preds[0].plate, "AB");
}

#[test]
fn test_postprocess_keeps_pad_char_when_requested() {
    let alphabet = "ABC_";
    let max_slots = 4;
    let vocab = 4;
    let indices = [0usize, 1, 3, 3]; // A B _ _
    let logits = fake_logits(max_slots, vocab, &indices);

    let preds = postprocess_output(
        &logits,
        1,
        max_slots,
        alphabet,
        '_',
        false, // keep pad
        false,
        None,
        None,
    )
    .unwrap();

    assert_eq!(preds[0].plate, "AB__");
}

#[test]
fn test_postprocess_confidence() {
    let alphabet = "ABC_";
    let max_slots = 3;
    let vocab = 4;
    let indices = [0usize, 1, 2]; // A B C
    let logits = fake_logits(max_slots, vocab, &indices);

    let preds = postprocess_output(
        &logits,
        1,
        max_slots,
        alphabet,
        '_',
        true,
        true, // return confidence
        None,
        None,
    )
    .unwrap();

    let probs = preds[0].char_probs.as_ref().unwrap();
    assert_eq!(probs.len(), max_slots);
    for &p in probs {
        assert!((p - 1.0).abs() < 1e-5, "expected max=1.0 from one-hot, got {p}");
    }
}

#[test]
fn test_postprocess_batch() {
    let alphabet = "ABC_";
    let max_slots = 2;
    let vocab = 4;
    let n = 3;

    let all: Vec<f32> = (0..n)
        .flat_map(|i| fake_logits(max_slots, vocab, &[i % vocab, (i + 1) % vocab]))
        .collect();

    let preds = postprocess_output(&all, n, max_slots, alphabet, '_', false, false, None, None)
        .unwrap();

    assert_eq!(preds.len(), 3);
}

#[test]
fn test_postprocess_wrong_length_errors() {
    let result = postprocess_output(
        &[0.1f32; 10], // wrong length
        1,
        7,    // max_slots
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_", // 37 chars → 7*37=259 expected
        '_',
        true,
        false,
        None,
        None,
    );
    assert!(result.is_err());
}

#[test]
fn test_postprocess_region() {
    let alphabet = "ABC_";
    let max_slots = 2;
    let vocab = 4;
    let plate_logits = fake_logits(max_slots, vocab, &[0, 1]);

    // Region output for 3 regions; best is index 2
    let region_out = vec![0.1f32, 0.2, 0.7];
    let region_labels = vec!["Argentina".to_owned(), "Brazil".to_owned(), "Chile".to_owned()];

    let preds = postprocess_output(
        &plate_logits,
        1,
        max_slots,
        alphabet,
        '_',
        true,
        true,
        Some(&region_out),
        Some(&region_labels),
    )
    .unwrap();

    assert_eq!(preds[0].region.as_deref(), Some("Chile"));
    let rp = preds[0].region_prob.unwrap();
    assert!((rp - 0.7).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// OcrModel hub parsing
// ---------------------------------------------------------------------------

#[test]
fn test_ocr_model_round_trip() {
    use fpo_rust::OcrModel;

    let models = [
        OcrModel::CctSV2Global,
        OcrModel::ArgentinianPlatesCnn,
        OcrModel::EuropeanPlatesMobileVitV2,
    ];
    for m in &models {
        let s = m.as_str();
        let parsed = OcrModel::from_str(s).expect("should parse back");
        assert_eq!(parsed, *m);
    }
}

#[test]
fn test_ocr_model_unknown_returns_none() {
    use fpo_rust::OcrModel;
    assert!(OcrModel::from_str("does-not-exist").is_none());
}
