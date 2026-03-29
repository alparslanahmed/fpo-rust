//! Image pre-processing and model-output post-processing.

use crate::config::{ImageColorMode, ImageInterpolation, PaddingColor, PlateConfig};
use anyhow::{bail, Context};
use image::{
    imageops::{self, FilterType},
    DynamicImage, GrayImage, ImageBuffer, Luma, Rgb, RgbImage,
};
use std::path::Path;

// ---------------------------------------------------------------------------
// Image reading
// ---------------------------------------------------------------------------

/// Read an image from disk in the colour mode specified by `config`.
///
/// Returns a `DynamicImage` whose colour space matches the config:
/// - `Grayscale` → `DynamicImage::ImageLuma8`
/// - `Rgb`       → `DynamicImage::ImageRgb8`
pub fn read_plate_image(
    path: impl AsRef<Path>,
    color_mode: &ImageColorMode,
) -> anyhow::Result<DynamicImage> {
    let img = image::open(path.as_ref())
        .with_context(|| format!("Cannot open image: {}", path.as_ref().display()))?;

    let out = match color_mode {
        ImageColorMode::Grayscale => DynamicImage::ImageLuma8(img.to_luma8()),
        ImageColorMode::Rgb => DynamicImage::ImageRgb8(img.to_rgb8()),
    };
    Ok(out)
}

// ---------------------------------------------------------------------------
// Interpolation helpers
// ---------------------------------------------------------------------------

fn to_filter(interp: &ImageInterpolation) -> FilterType {
    match interp {
        ImageInterpolation::Nearest => FilterType::Nearest,
        ImageInterpolation::Linear => FilterType::Triangle,
        ImageInterpolation::Cubic => FilterType::CatmullRom,
        ImageInterpolation::Area => FilterType::Lanczos3, // no "area" in `image`
        ImageInterpolation::Lanczos4 => FilterType::Lanczos3,
    }
}

// ---------------------------------------------------------------------------
// Resizing
// ---------------------------------------------------------------------------

/// Resize a dynamic image to `(target_w, target_h)`, honouring the config options.
///
/// When `keep_aspect_ratio` is true the image is letter-boxed with `padding_color`.
/// The output always has the colour space implied by `color_mode`.
pub fn resize_image(
    img: DynamicImage,
    target_h: u32,
    target_w: u32,
    color_mode: &ImageColorMode,
    keep_aspect_ratio: bool,
    interp: &ImageInterpolation,
    padding_color: &PaddingColor,
) -> anyhow::Result<DynamicImage> {
    let filter = to_filter(interp);

    if !keep_aspect_ratio {
        let resized = img.resize_exact(target_w, target_h, filter);
        return Ok(match color_mode {
            ImageColorMode::Grayscale => DynamicImage::ImageLuma8(resized.to_luma8()),
            ImageColorMode::Rgb => DynamicImage::ImageRgb8(resized.to_rgb8()),
        });
    }

    // --- letter-box ---
    let (orig_w, orig_h) = (img.width(), img.height());
    let scale = (target_w as f64 / orig_w as f64).min(target_h as f64 / orig_h as f64);
    let new_w = (orig_w as f64 * scale).round() as u32;
    let new_h = (orig_h as f64 * scale).round() as u32;

    let resized = img.resize_exact(new_w, new_h, filter);

    let pad_left = ((target_w - new_w) as f64 / 2.0 - 0.1).round() as u32;
    let pad_top = ((target_h - new_h) as f64 / 2.0 - 0.1).round() as u32;

    match color_mode {
        ImageColorMode::Grayscale => {
            let fill = Luma([padding_color.as_gray()]);
            let mut canvas: GrayImage = ImageBuffer::from_pixel(target_w, target_h, fill);
            imageops::overlay(&mut canvas, &resized.to_luma8(), pad_left as i64, pad_top as i64);
            Ok(DynamicImage::ImageLuma8(canvas))
        }
        ImageColorMode::Rgb => {
            let [r, g, b] = padding_color.as_rgb();
            let fill = Rgb([r, g, b]);
            let mut canvas: RgbImage = ImageBuffer::from_pixel(target_w, target_h, fill);
            imageops::overlay(&mut canvas, &resized.to_rgb8(), pad_left as i64, pad_top as i64);
            Ok(DynamicImage::ImageRgb8(canvas))
        }
    }
}

/// Convenience wrapper: read an image from disk and resize it.
pub fn read_and_resize_plate_image(
    path: impl AsRef<Path>,
    cfg: &PlateConfig,
) -> anyhow::Result<DynamicImage> {
    let img = read_plate_image(path, &cfg.image_color_mode)?;
    resize_image(
        img,
        cfg.img_height,
        cfg.img_width,
        &cfg.image_color_mode,
        cfg.keep_aspect_ratio,
        &cfg.interpolation,
        &cfg.padding_color,
    )
}

// ---------------------------------------------------------------------------
// Pre-processing: image(s) → flat u8 tensor (N, H, W, C)
// ---------------------------------------------------------------------------

/// Convert a `DynamicImage` into a flat `Vec<u8>` in `(H, W, C)` order.
pub fn image_to_hwc(img: &DynamicImage, color_mode: &ImageColorMode) -> Vec<u8> {
    match color_mode {
        ImageColorMode::Grayscale => img.to_luma8().into_raw(),
        ImageColorMode::Rgb => img.to_rgb8().into_raw(),
    }
}

/// Build the ONNX-ready `u8` tensor `(N, H, W, C)` from a batch of images.
///
/// Each image must already have the correct `(H, W)` dimensions.
pub fn images_to_batch(imgs: &[DynamicImage], cfg: &PlateConfig) -> Vec<u8> {
    imgs.iter()
        .flat_map(|img| image_to_hwc(img, &cfg.image_color_mode))
        .collect()
}

// ---------------------------------------------------------------------------
// Post-processing: raw model output → predictions
// ---------------------------------------------------------------------------

/// Per-image plate prediction returned by the recognizer.
#[derive(Debug, Clone)]
pub struct PlatePrediction {
    /// Decoded license-plate string.
    pub plate: String,
    /// Per-character confidence scores (present when `return_confidence = true`).
    pub char_probs: Option<Vec<f32>>,
    /// Predicted region / country label (present when the model has a region head).
    pub region: Option<String>,
    /// Probability for the predicted region (present together with `region` when
    /// `return_confidence = true`).
    pub region_prob: Option<f32>,
}

/// Decode the raw plate-head output tensor into `PlatePrediction` values.
///
/// # Parameters
/// * `model_output`     – flat f32 slice of shape `(N * max_plate_slots * vocab_size)`.
/// * `n`                – batch size.
/// * `max_plate_slots`  – number of character positions.
/// * `alphabet`         – the model's character set.
/// * `pad_char`         – padding character to strip from the right.
/// * `remove_pad_char`  – whether to strip trailing `pad_char`.
/// * `return_confidence`– include per-character probabilities.
/// * `region_output`    – optional flat f32 slice `(N * num_regions)` (already softmaxed).
/// * `region_labels`    – label list for the region head.
pub fn postprocess_output(
    model_output: &[f32],
    n: usize,
    max_plate_slots: usize,
    alphabet: &str,
    pad_char: char,
    remove_pad_char: bool,
    return_confidence: bool,
    region_output: Option<&[f32]>,
    region_labels: Option<&[String]>,
) -> anyhow::Result<Vec<PlatePrediction>> {
    let vocab_size = alphabet.chars().count();
    if model_output.len() != n * max_plate_slots * vocab_size {
        bail!(
            "Unexpected model output length: got {}, expected {} (n={n}, slots={max_plate_slots}, vocab={vocab_size})",
            model_output.len(),
            n * max_plate_slots * vocab_size
        );
    }

    let chars: Vec<char> = alphabet.chars().collect();
    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        let sample = &model_output[i * max_plate_slots * vocab_size..(i + 1) * max_plate_slots * vocab_size];

        let mut plate = String::with_capacity(max_plate_slots);
        let mut probs = if return_confidence {
            Some(Vec::with_capacity(max_plate_slots))
        } else {
            None
        };

        for slot in 0..max_plate_slots {
            let logits = &sample[slot * vocab_size..(slot + 1) * vocab_size];
            let (best_idx, &best_val) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            plate.push(chars[best_idx]);
            if let Some(ref mut p) = probs {
                p.push(best_val);
            }
        }

        if remove_pad_char {
            while plate.ends_with(pad_char) {
                plate.pop();
            }
        }

        // Region
        let (region, region_prob) = match (region_output, region_labels) {
            (Some(ro), Some(rl)) => {
                let num_regions = rl.len();
                if num_regions == 0 {
                    (None, None)
                } else {
                    let rsample = &ro[i * num_regions..(i + 1) * num_regions];
                    let (ridx, &rval) = rsample
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap();
                    let label = rl.get(ridx).map(|s| s.clone());
                    let prob = if return_confidence { Some(rval) } else { None };
                    (label, prob)
                }
            }
            _ => (None, None),
        };

        results.push(PlatePrediction {
            plate,
            char_probs: probs,
            region,
            region_prob,
        });
    }

    Ok(results)
}
