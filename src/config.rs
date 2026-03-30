//! Plate configuration for inference, parsed from a YAML file.

use anyhow::Context;
use serde::Deserialize;
use std::path::Path;

/// Interpolation method used when resizing the plate image.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ImageInterpolation {
    Nearest,
    #[default]
    Linear,
    Cubic,
    Area,
    Lanczos4,
}

/// Colour mode of the plate image fed into the model.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ImageColorMode {
    #[default]
    Grayscale,
    Rgb,
}

/// Padding colour used during letter-box resizing.
///
/// YAML accepts a single integer (grayscale) or a three-element sequence [r, g, b].
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum PaddingColor {
    Gray(u8),
    Rgb([u8; 3]),
}

impl Default for PaddingColor {
    fn default() -> Self {
        PaddingColor::Rgb([114, 114, 114])
    }
}

impl PaddingColor {
    /// Return the grayscale byte value (uses the first channel for RGB).
    pub fn as_gray(&self) -> u8 {
        match self {
            PaddingColor::Gray(v) => *v,
            PaddingColor::Rgb([r, _, _]) => *r,
        }
    }

    /// Return the RGB byte triple.
    pub fn as_rgb(&self) -> [u8; 3] {
        match self {
            PaddingColor::Gray(v) => [*v, *v, *v],
            PaddingColor::Rgb(c) => *c,
        }
    }
}

/// Inference configuration for a plate-recognition model, read from a YAML file.
#[derive(Debug, Clone, Deserialize)]
pub struct PlateConfig {
    /// Maximum number of character slots predicted by the model.
    pub max_plate_slots: usize,
    /// Full alphabet used by the model (one character per class).
    pub alphabet: String,
    /// Padding character appended to plates shorter than `max_plate_slots`.
    pub pad_char: char,
    /// Image height the model expects.
    pub img_height: u32,
    /// Image width the model expects.
    pub img_width: u32,
    /// Whether to preserve aspect ratio via letter-box padding.
    #[serde(default)]
    pub keep_aspect_ratio: bool,
    /// Interpolation method used when resizing.
    #[serde(default)]
    pub interpolation: ImageInterpolation,
    /// Colour mode: grayscale (1-channel) or rgb (3-channel).
    #[serde(default)]
    pub image_color_mode: ImageColorMode,
    /// Padding colour used when `keep_aspect_ratio` is true.
    #[serde(default)]
    pub padding_color: PaddingColor,
    /// Optional list of region / country labels the model can predict.
    #[serde(default)]
    pub plate_regions: Option<Vec<String>>,
}

impl PlateConfig {
    /// Load a `PlateConfig` from a YAML file.
    pub fn from_yaml(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Cannot read config: {}", path.as_ref().display()))?;
        let cfg: PlateConfig = serde_yml::from_str(&text)
            .with_context(|| format!("Cannot parse config: {}", path.as_ref().display()))?;
        Ok(cfg)
    }

    /// Number of colour channels (1 for grayscale, 3 for RGB).
    pub fn num_channels(&self) -> u32 {
        match self.image_color_mode {
            ImageColorMode::Rgb => 3,
            ImageColorMode::Grayscale => 1,
        }
    }

    /// `true` if the model includes a region-recognition head.
    pub fn has_region_recognition(&self) -> bool {
        self.plate_regions
            .as_ref()
            .map_or(false, |v| !v.is_empty())
    }

    /// Index of `pad_char` in the alphabet.
    pub fn pad_idx(&self) -> usize {
        self.alphabet
            .chars()
            .position(|c| c == self.pad_char)
            .unwrap_or(0)
    }
}
