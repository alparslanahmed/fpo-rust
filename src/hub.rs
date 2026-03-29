//! Model hub: known ONNX models with their download URLs and local caching.

use anyhow::{bail, Context};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
};

/// The set of pre-trained OCR models available for download.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OcrModel {
    /// Compact Convolutional Transformer – Small, v2, global plates.
    CctSV2Global,
    /// Compact Convolutional Transformer – XSmall, v2, global plates.
    CctXsV2Global,
    /// Compact Convolutional Transformer – Small, v1, global plates.
    CctSV1Global,
    /// Compact Convolutional Transformer – XSmall, v1, global plates.
    CctXsV1Global,
    /// Compact Convolutional Transformer – Small, ReLU, v1, global plates.
    CctSReluV1Global,
    /// Compact Convolutional Transformer – XSmall, ReLU, v1, global plates.
    CctXsReluV1Global,
    /// Argentinian plates CNN model.
    ArgentinianPlatesCnn,
    /// Argentinian plates CNN model trained with synthetic data.
    ArgentinianPlatesCnnSynth,
    /// European plates MobileVIT-v2 model.
    EuropeanPlatesMobileVitV2,
    /// Global plates (65+ countries) MobileVIT-v2 model.
    GlobalPlatesMobileVitV2,
}

impl OcrModel {
    /// Return the string identifier used to name cache directories.
    pub fn as_str(&self) -> &'static str {
        match self {
            OcrModel::CctSV2Global => "cct-s-v2-global-model",
            OcrModel::CctXsV2Global => "cct-xs-v2-global-model",
            OcrModel::CctSV1Global => "cct-s-v1-global-model",
            OcrModel::CctXsV1Global => "cct-xs-v1-global-model",
            OcrModel::CctSReluV1Global => "cct-s-relu-v1-global-model",
            OcrModel::CctXsReluV1Global => "cct-xs-relu-v1-global-model",
            OcrModel::ArgentinianPlatesCnn => "argentinian-plates-cnn-model",
            OcrModel::ArgentinianPlatesCnnSynth => "argentinian-plates-cnn-synth-model",
            OcrModel::EuropeanPlatesMobileVitV2 => "european-plates-mobile-vit-v2-model",
            OcrModel::GlobalPlatesMobileVitV2 => "global-plates-mobile-vit-v2-model",
        }
    }

    /// Return `(onnx_url, config_url)` for this model.
    pub fn urls(&self) -> (&'static str, &'static str) {
        match self {
            OcrModel::CctSV2Global => (
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/cct_s_v2_global.onnx"
                ),
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/cct_s_v2_global_plate_config.yaml"
                ),
            ),
            OcrModel::CctXsV2Global => (
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/cct_xs_v2_global.onnx"
                ),
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/cct_xs_v2_global_plate_config.yaml"
                ),
            ),
            OcrModel::CctSV1Global => (
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/cct_s_v1_global.onnx"
                ),
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/cct_s_v1_global_plate_config.yaml"
                ),
            ),
            OcrModel::CctXsV1Global => (
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/cct_xs_v1_global.onnx"
                ),
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/cct_xs_v1_global_plate_config.yaml"
                ),
            ),
            OcrModel::CctSReluV1Global => (
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/cct_s_relu_v1_global.onnx"
                ),
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/cct_s_relu_v1_global_plate_config.yaml"
                ),
            ),
            OcrModel::CctXsReluV1Global => (
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/cct_xs_relu_v1_global.onnx"
                ),
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/cct_xs_relu_v1_global_plate_config.yaml"
                ),
            ),
            OcrModel::ArgentinianPlatesCnn => (
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/arg_cnn_ocr.onnx"
                ),
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/arg_cnn_ocr_config.yaml"
                ),
            ),
            OcrModel::ArgentinianPlatesCnnSynth => (
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/arg_cnn_ocr_synth.onnx"
                ),
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/arg_cnn_ocr_config.yaml"
                ),
            ),
            OcrModel::EuropeanPlatesMobileVitV2 => (
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/european_mobile_vit_v2_ocr.onnx"
                ),
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/european_mobile_vit_v2_ocr_config.yaml"
                ),
            ),
            OcrModel::GlobalPlatesMobileVitV2 => (
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/global_mobile_vit_v2_ocr.onnx"
                ),
                concat!(
                    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/",
                    "arg-plates/global_mobile_vit_v2_ocr_config.yaml"
                ),
            ),
        }
    }

    /// Parse a model from its string identifier.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "cct-s-v2-global-model" => Some(OcrModel::CctSV2Global),
            "cct-xs-v2-global-model" => Some(OcrModel::CctXsV2Global),
            "cct-s-v1-global-model" => Some(OcrModel::CctSV1Global),
            "cct-xs-v1-global-model" => Some(OcrModel::CctXsV1Global),
            "cct-s-relu-v1-global-model" => Some(OcrModel::CctSReluV1Global),
            "cct-xs-relu-v1-global-model" => Some(OcrModel::CctXsReluV1Global),
            "argentinian-plates-cnn-model" => Some(OcrModel::ArgentinianPlatesCnn),
            "argentinian-plates-cnn-synth-model" => Some(OcrModel::ArgentinianPlatesCnnSynth),
            "european-plates-mobile-vit-v2-model" => Some(OcrModel::EuropeanPlatesMobileVitV2),
            "global-plates-mobile-vit-v2-model" => Some(OcrModel::GlobalPlatesMobileVitV2),
            _ => None,
        }
    }
}

/// Default cache directory: `~/.cache/fast-plate-ocr/`.
pub fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("fast-plate-ocr")
}

/// Download a single file from `url` to `dest`, showing a progress bar.
fn download_file(url: &str, dest: &Path) -> anyhow::Result<()> {
    let mut response = ureq::get(url)
        .call()
        .with_context(|| format!("HTTP request failed for {url}"))?;

    let content_length = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    let file_name = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| url.to_owned());

    let pb = ProgressBar::new(content_length);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
        )
        .unwrap()
        .progress_chars("##-"),
    );
    pb.set_message(format!("Downloading {file_name}"));

    // Write to a temporary sibling file then rename (atomic-ish).
    let tmp = dest.with_extension("tmp");
    {
        let mut file =
            std::fs::File::create(&tmp).with_context(|| format!("Cannot create {}", tmp.display()))?;

        let mut buf = [0u8; 65_536];
        let body = response.body_mut();
        loop {
            let n = body
                .as_reader()
                .read(&mut buf)
                .context("Error reading HTTP body")?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).context("Error writing file")?;
            pb.inc(n as u64);
        }
    }
    pb.finish_with_message(format!("Saved {file_name}"));

    std::fs::rename(&tmp, dest)
        .with_context(|| format!("Cannot rename {} → {}", tmp.display(), dest.display()))?;

    Ok(())
}

/// Download an OCR model from the hub and return `(onnx_path, config_path)`.
///
/// Files are cached in `save_dir` (defaults to `~/.cache/fast-plate-ocr/<model_name>/`).
/// Set `force_download = true` to re-download even if the files already exist.
pub fn download_model(
    model: &OcrModel,
    save_dir: Option<&Path>,
    force_download: bool,
) -> anyhow::Result<(PathBuf, PathBuf)> {
    let cache_dir = match save_dir {
        Some(d) => d.to_path_buf(),
        None => default_cache_dir().join(model.as_str()),
    };

    if cache_dir.is_file() {
        bail!("Expected a directory but found a file: {}", cache_dir.display());
    }

    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("Cannot create cache dir {}", cache_dir.display()))?;

    let (model_url, config_url) = model.urls();

    let model_filename = cache_dir.join(
        model_url
            .rsplit('/')
            .next()
            .expect("URL must have a path segment"),
    );
    let config_filename = cache_dir.join(
        config_url
            .rsplit('/')
            .next()
            .expect("URL must have a path segment"),
    );

    if force_download || !model_filename.is_file() {
        download_file(model_url, &model_filename)?;
    }
    if force_download || !config_filename.is_file() {
        download_file(config_url, &config_filename)?;
    }

    Ok((model_filename, config_filename))
}
