//! License-plate recognizer with selectable inference backends.

use crate::{
    config::PlateConfig,
    hub::{download_model, OcrModel},
    process::{images_to_batch, postprocess_output, read_and_resize_plate_image, PlatePrediction},
};
use anyhow::{bail, Context};
use image::DynamicImage;
use std::{path::Path, time::Instant};
use tract_onnx::prelude::*;

// ---------------------------------------------------------------------------
// Input type
// ---------------------------------------------------------------------------

/// A single plate input: either a file path or an already-decoded image.
pub enum PlateInput<'a> {
    Path(&'a Path),
    Image(DynamicImage),
}

impl<'a> From<&'a str> for PlateInput<'a> {
    fn from(s: &'a str) -> Self {
        PlateInput::Path(Path::new(s))
    }
}

impl<'a> From<&'a Path> for PlateInput<'a> {
    fn from(p: &'a Path) -> Self {
        PlateInput::Path(p)
    }
}

impl From<DynamicImage> for PlateInput<'_> {
    fn from(img: DynamicImage) -> Self {
        PlateInput::Image(img)
    }
}

// ---------------------------------------------------------------------------
// Recognizer
// ---------------------------------------------------------------------------

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

struct TractBackend {
    model: OnnxModel,
    plate_output_idx: usize,
    region_output_idx: Option<usize>,
}

enum InferenceBackend {
    Tract(TractBackend),
    #[cfg(feature = "ncnn")]
    Ncnn(crate::ncnn_backend::NcnnBackend),
}

/// Inference class for license-plate OCR.
pub struct LicensePlateRecognizer {
    backend: InferenceBackend,
    /// Plate configuration (image size, alphabet, etc.).
    pub config: PlateConfig,
    /// Human-readable model name.
    pub model_name: String,
    /// `true` when the model has a region head and the config defines regions.
    has_region_head: bool,
}

impl LicensePlateRecognizer {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Build a recognizer from an existing ONNX model file and config file.
    pub fn from_files(
        onnx_model_path: impl AsRef<Path>,
        plate_config_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        let onnx_path = onnx_model_path.as_ref();
        let cfg_path = plate_config_path.as_ref();

        if !onnx_path.exists() {
            bail!("ONNX model not found: {}", onnx_path.display());
        }
        if !cfg_path.exists() {
            bail!("Plate config not found: {}", cfg_path.display());
        }

        let model_name = onnx_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "custom".to_owned());

        let config = PlateConfig::from_yaml(cfg_path)?;

        let model = Self::load_model(onnx_path, &config)?;

        Self::from_tract_model_and_config(model, config, model_name)
    }

    /// Download (if necessary) and load a model from the hub.
    pub fn from_hub(model: OcrModel, force_download: bool) -> anyhow::Result<Self> {
        let model_name = model.as_str().to_owned();
        let (onnx_path, cfg_path) = download_model(&model, None, force_download)?;
        let mut recognizer = Self::from_files(onnx_path, cfg_path)?;
        recognizer.model_name = model_name;
        Ok(recognizer)
    }

    /// Download (if necessary) and load a model from the hub, saving to a specific directory.
    pub fn from_hub_to_dir(
        model: OcrModel,
        save_dir: &Path,
        force_download: bool,
    ) -> anyhow::Result<Self> {
        let model_name = model.as_str().to_owned();
        let (onnx_path, cfg_path) = download_model(&model, Some(save_dir), force_download)?;
        let mut recognizer = Self::from_files(onnx_path, cfg_path)?;
        recognizer.model_name = model_name;
        Ok(recognizer)
    }

    /// Download a hub model config and load an already-converted NCNN model from the same cache
    /// directory.
    ///
    /// The hub distributes ONNX files. Run `fpo-rust convert-ncnn --model <NAME>` first to create
    /// the matching `.ncnn.param` and `.ncnn.bin` files.
    #[cfg(feature = "ncnn")]
    pub fn from_hub_ncnn(model: OcrModel, force_download: bool) -> anyhow::Result<Self> {
        Self::from_hub_ncnn_with_options(
            model,
            force_download,
            crate::ncnn_backend::NcnnOptions::default(),
        )
    }

    /// Download a hub model config and load converted NCNN files with explicit runtime options.
    #[cfg(feature = "ncnn")]
    pub fn from_hub_ncnn_with_options(
        model: OcrModel,
        force_download: bool,
        options: crate::ncnn_backend::NcnnOptions,
    ) -> anyhow::Result<Self> {
        let model_name = model.as_str().to_owned();
        let (onnx_path, cfg_path) = download_model(&model, None, force_download)?;
        let (param_path, bin_path) = crate::ncnn_backend::ncnn_paths_for_onnx(&onnx_path)?;

        if !param_path.is_file() || !bin_path.is_file() {
            bail!(
                "NCNN files not found for hub model '{model_name}'. Run: fpo-rust convert-ncnn --model {model_name}"
            );
        }

        let mut recognizer =
            Self::from_ncnn_files_with_options(param_path, bin_path, cfg_path, options)?;
        recognizer.model_name = format!("{model_name} (ncnn)");
        Ok(recognizer)
    }

    /// Download a hub model config to a specific directory and load an already-converted NCNN
    /// model from that directory.
    #[cfg(feature = "ncnn")]
    pub fn from_hub_to_dir_ncnn(
        model: OcrModel,
        save_dir: &Path,
        force_download: bool,
    ) -> anyhow::Result<Self> {
        Self::from_hub_to_dir_ncnn_with_options(
            model,
            save_dir,
            force_download,
            crate::ncnn_backend::NcnnOptions::default(),
        )
    }

    /// Download a hub model config to a specific directory and load converted NCNN files with
    /// explicit runtime options.
    #[cfg(feature = "ncnn")]
    pub fn from_hub_to_dir_ncnn_with_options(
        model: OcrModel,
        save_dir: &Path,
        force_download: bool,
        options: crate::ncnn_backend::NcnnOptions,
    ) -> anyhow::Result<Self> {
        let model_name = model.as_str().to_owned();
        let (onnx_path, cfg_path) = download_model(&model, Some(save_dir), force_download)?;
        let (param_path, bin_path) = crate::ncnn_backend::ncnn_paths_for_onnx(&onnx_path)?;

        if !param_path.is_file() || !bin_path.is_file() {
            bail!(
                "NCNN files not found for hub model '{model_name}' in {}. Run: fpo-rust convert-ncnn --model {model_name} --out-dir {}",
                save_dir.display(),
                save_dir.display()
            );
        }

        let mut recognizer =
            Self::from_ncnn_files_with_options(param_path, bin_path, cfg_path, options)?;
        recognizer.model_name = format!("{model_name} (ncnn)");
        Ok(recognizer)
    }

    /// Build a recognizer from an NCNN `.param`/`.bin` pair and a plate config.
    #[cfg(feature = "ncnn")]
    pub fn from_ncnn_files(
        ncnn_param_path: impl AsRef<Path>,
        ncnn_bin_path: impl AsRef<Path>,
        plate_config_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        Self::from_ncnn_files_with_options(
            ncnn_param_path,
            ncnn_bin_path,
            plate_config_path,
            crate::ncnn_backend::NcnnOptions::default(),
        )
    }

    /// Build a recognizer from an NCNN `.param`/`.bin` pair with explicit runtime options.
    #[cfg(feature = "ncnn")]
    pub fn from_ncnn_files_with_options(
        ncnn_param_path: impl AsRef<Path>,
        ncnn_bin_path: impl AsRef<Path>,
        plate_config_path: impl AsRef<Path>,
        options: crate::ncnn_backend::NcnnOptions,
    ) -> anyhow::Result<Self> {
        let param_path = ncnn_param_path.as_ref();
        let bin_path = ncnn_bin_path.as_ref();
        let cfg_path = plate_config_path.as_ref();

        if !cfg_path.exists() {
            bail!("Plate config not found: {}", cfg_path.display());
        }

        let model_name = param_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "custom-ncnn".to_owned());
        let config = PlateConfig::from_yaml(cfg_path)?;
        let ncnn =
            crate::ncnn_backend::NcnnBackend::from_files(param_path, bin_path, &config, options)?;
        let has_region_head = ncnn.has_region_output() && config.has_region_recognition();

        if !has_region_head && config.has_region_recognition() {
            eprintln!(
                "Warning: plate config declares regions but no NCNN region output is configured. \
                 Region predictions will be disabled."
            );
        }

        Ok(Self {
            backend: InferenceBackend::Ncnn(ncnn),
            config,
            model_name,
            has_region_head,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn load_model(onnx_path: &Path, config: &PlateConfig) -> anyhow::Result<OnnxModel> {
        let h = config.img_height;
        let w = config.img_width;
        let c = config.num_channels();

        let plan = tract_onnx::onnx()
            .model_for_path(onnx_path)
            .context("Cannot parse ONNX model")?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    u8::datum_type(),
                    tvec![1usize, h as usize, w as usize, c as usize],
                ),
            )
            .context("Cannot set input fact")?
            .into_optimized()
            .context("Cannot optimise ONNX model")?
            .into_runnable()
            .context("Cannot make model runnable")?;

        Ok(plan)
    }

    fn from_tract_model_and_config(
        model: OnnxModel,
        config: PlateConfig,
        model_name: String,
    ) -> anyhow::Result<Self> {
        // tract does not expose named outputs by default; we rely on output index ordering.
        // By convention in fast-plate-ocr ONNX exports:
        //   output 0 = plate logits
        //   output 1 = region logits (only when the model has a region head)
        let num_outputs = model.model().output_outlets()?.len();
        let plate_output_idx = 0;
        let region_output_idx = if num_outputs > 1 { Some(1) } else { None };

        let has_region_head = region_output_idx.is_some() && config.has_region_recognition();

        if region_output_idx.is_none() && config.has_region_recognition() {
            eprintln!(
                "Warning: plate config declares regions but the model has only one output. \
                 Region predictions will be disabled."
            );
        }
        if region_output_idx.is_some() && !config.has_region_recognition() {
            eprintln!(
                "Warning: model has a second output but the plate config has no region list. \
                 Region predictions will be disabled."
            );
        }

        Ok(Self {
            backend: InferenceBackend::Tract(TractBackend {
                model,
                plate_output_idx,
                region_output_idx,
            }),
            config,
            model_name,
            has_region_head,
        })
    }

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /// Run plate recognition on one or more plate images.
    ///
    /// `inputs` is a slice of `PlateInput` (paths or pre-loaded images).  
    /// Returns one `PlatePrediction` per input.
    pub fn run(
        &self,
        inputs: &[PlateInput<'_>],
        return_confidence: bool,
        remove_pad_char: bool,
    ) -> anyhow::Result<Vec<PlatePrediction>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }

        // --- Load & resize images ---
        let imgs: Vec<DynamicImage> = inputs
            .iter()
            .map(|inp| match inp {
                PlateInput::Path(p) => read_and_resize_plate_image(p, &self.config),
                PlateInput::Image(img) => crate::process::resize_image(
                    img.clone(),
                    self.config.img_height,
                    self.config.img_width,
                    &self.config.image_color_mode,
                    self.config.keep_aspect_ratio,
                    &self.config.interpolation,
                    &self.config.padding_color,
                ),
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        // Run images one at a time (tract handles 1-image batches well; for larger batches
        // we iterate and concatenate postprocessing results).
        let mut plate_data_all: Vec<f32> = Vec::new();
        let mut region_data_all: Vec<f32> = Vec::new();
        let n = imgs.len();

        for img in &imgs {
            let raw = images_to_batch(std::slice::from_ref(img), &self.config);
            let (plate_data, region_data) = self.run_model_raw(&raw, self.has_region_head)?;
            plate_data_all.extend_from_slice(&plate_data);
            if let Some(region_data) = region_data {
                region_data_all.extend_from_slice(&region_data);
            }
        }

        postprocess_output(
            &plate_data_all,
            n,
            self.config.max_plate_slots,
            &self.config.alphabet,
            self.config.pad_char,
            remove_pad_char,
            return_confidence,
            if self.has_region_head && !region_data_all.is_empty() {
                Some(&region_data_all)
            } else {
                None
            },
            if self.has_region_head {
                self.config.plate_regions.as_deref()
            } else {
                None
            },
        )
    }

    /// Convenience wrapper for a single image.
    pub fn run_one(
        &self,
        input: PlateInput<'_>,
        return_confidence: bool,
        remove_pad_char: bool,
    ) -> anyhow::Result<PlatePrediction> {
        let mut results = self.run(&[input], return_confidence, remove_pad_char)?;
        if results.len() != 1 {
            bail!("Expected exactly 1 result, got {}", results.len());
        }
        Ok(results.remove(0))
    }

    fn run_model_raw(
        &self,
        raw: &[u8],
        with_region: bool,
    ) -> anyhow::Result<(Vec<f32>, Option<Vec<f32>>)> {
        match &self.backend {
            InferenceBackend::Tract(backend) => {
                let h = self.config.img_height as usize;
                let w = self.config.img_width as usize;
                let c = self.config.num_channels() as usize;

                let input_tensor: Tensor =
                    tract_ndarray::Array4::<u8>::from_shape_vec((1, h, w, c), raw.to_vec())
                        .context("Cannot build input array")?
                        .into();

                let outputs = backend
                    .model
                    .run(tvec![input_tensor.into()])
                    .context("Model run failed")?;

                let plate_out = outputs
                    .get(backend.plate_output_idx)
                    .context("Missing plate output")?;
                let plate_view = plate_out
                    .to_array_view::<f32>()
                    .context("Cannot read plate output as f32")?;
                let plate_data = plate_view
                    .as_slice()
                    .context("Plate output is not contiguous")?
                    .to_vec();

                let region_data = if with_region {
                    if let Some(ridx) = backend.region_output_idx {
                        if let Some(region_out) = outputs.get(ridx) {
                            let region_view = region_out
                                .to_array_view::<f32>()
                                .context("Cannot read region output as f32")?;
                            Some(
                                region_view
                                    .as_slice()
                                    .context("Region output is not contiguous")?
                                    .to_vec(),
                            )
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                Ok((plate_data, region_data))
            }
            #[cfg(feature = "ncnn")]
            InferenceBackend::Ncnn(backend) => backend.run_raw(raw, &self.config, with_region),
        }
    }

    // -----------------------------------------------------------------------
    // Benchmark
    // -----------------------------------------------------------------------

    /// Run a simple throughput benchmark and print the results to stdout.
    ///
    /// * `n_iter`   – number of timed iterations
    /// * `batch_size` – images per "batch" (each is run individually due to tract's fixed shape)
    /// * `warmup`   – number of warm-up iterations (not counted)
    /// * `include_processing` – if `true`, preprocessing / postprocessing are included in the timing.
    pub fn benchmark(
        &self,
        n_iter: usize,
        batch_size: usize,
        warmup: usize,
        include_processing: bool,
    ) -> anyhow::Result<()> {
        use image::{DynamicImage, ImageBuffer, Luma, Rgb};

        let h = self.config.img_height;
        let w = self.config.img_width;
        let c = self.config.num_channels();

        let raw_pixels: Vec<u8> = (0..(h as usize * w as usize * c as usize))
            .map(|i| (i % 256) as u8)
            .collect();

        let make_image = || -> DynamicImage {
            if c == 1 {
                DynamicImage::ImageLuma8(
                    ImageBuffer::<Luma<u8>, _>::from_raw(w, h, raw_pixels.clone()).unwrap(),
                )
            } else {
                DynamicImage::ImageRgb8(
                    ImageBuffer::<Rgb<u8>, _>::from_raw(w, h, raw_pixels.clone()).unwrap(),
                )
            }
        };

        let run_once = || -> anyhow::Result<()> {
            let img = make_image();
            if include_processing {
                let inputs = vec![PlateInput::Image(img)];
                self.run(&inputs, false, true)?;
            } else {
                let raw = images_to_batch(std::slice::from_ref(&img), &self.config);
                self.run_model_raw(&raw, false).context("benchmark run")?;
            }
            Ok(())
        };

        // Warm-up
        for _ in 0..warmup {
            run_once()?;
        }

        // Timed loop
        let t0 = Instant::now();
        for _ in 0..(n_iter * batch_size) {
            run_once()?;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;

        let total_plates = n_iter * batch_size;
        let avg_ms = if total_plates > 0 {
            elapsed_ms / n_iter as f64
        } else {
            0.0
        };
        let pps = if avg_ms > 0.0 {
            (1_000.0 / avg_ms) * batch_size as f64
        } else {
            0.0
        };

        println!("─────────────────────────────────────────");
        println!(" Model         : {}", self.model_name);
        println!(" Batch size    : {batch_size}");
        println!(" Warm-up iters : {warmup}");
        println!(" Timed iters   : {n_iter}");
        println!(" Avg time/batch: {avg_ms:.4} ms");
        println!(" Plates/second : {pps:.2}");
        println!("─────────────────────────────────────────");

        Ok(())
    }
}
