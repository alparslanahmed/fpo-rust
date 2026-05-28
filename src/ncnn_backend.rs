//! Tencent NCNN inference backend.
//!
//! The NCNN runtime consumes `.param` and `.bin` files produced by `pnnx`.

use crate::config::PlateConfig;
use anyhow::{bail, Context};
use std::{
    collections::HashSet,
    ffi::CString,
    os::raw::c_void,
    path::{Path, PathBuf},
};

mod ffi {
    use std::os::raw::{c_char, c_float, c_int, c_void};

    #[repr(C)]
    pub struct NcnnOption {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct NcnnMat {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct NcnnNet {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct NcnnExtractor {
        _private: [u8; 0],
    }

    pub type NcnnOptionT = *mut NcnnOption;
    pub type NcnnMatT = *mut NcnnMat;
    pub type NcnnNetT = *mut NcnnNet;
    pub type NcnnExtractorT = *mut NcnnExtractor;

    extern "C" {
        pub fn ncnn_option_create() -> NcnnOptionT;
        pub fn ncnn_option_destroy(opt: NcnnOptionT);
        pub fn ncnn_option_set_num_threads(opt: NcnnOptionT, num_threads: c_int);
        #[cfg(not(feature = "ncnn-cpu"))]
        pub fn ncnn_option_set_use_vulkan_compute(opt: NcnnOptionT, enable: c_int);

        pub fn ncnn_net_create() -> NcnnNetT;
        pub fn ncnn_net_destroy(net: NcnnNetT);
        pub fn ncnn_net_set_option(net: NcnnNetT, opt: NcnnOptionT);
        pub fn ncnn_net_load_param(net: NcnnNetT, path: *const c_char) -> c_int;
        pub fn ncnn_net_load_model(net: NcnnNetT, path: *const c_char) -> c_int;

        pub fn ncnn_extractor_create(net: NcnnNetT) -> NcnnExtractorT;
        pub fn ncnn_extractor_destroy(ex: NcnnExtractorT);
        pub fn ncnn_extractor_input(
            ex: NcnnExtractorT,
            name: *const c_char,
            mat: NcnnMatT,
        ) -> c_int;
        pub fn ncnn_extractor_extract(
            ex: NcnnExtractorT,
            name: *const c_char,
            mat: *mut NcnnMatT,
        ) -> c_int;

        pub fn ncnn_mat_create() -> NcnnMatT;
        pub fn ncnn_mat_create_external_3d(
            w: c_int,
            h: c_int,
            c: c_int,
            data: *mut c_void,
            allocator: *mut c_void,
        ) -> NcnnMatT;
        pub fn ncnn_mat_destroy(mat: NcnnMatT);
        pub fn ncnn_mat_get_dims(mat: NcnnMatT) -> c_int;
        pub fn ncnn_mat_get_w(mat: NcnnMatT) -> c_int;
        pub fn ncnn_mat_get_h(mat: NcnnMatT) -> c_int;
        pub fn ncnn_mat_get_d(mat: NcnnMatT) -> c_int;
        pub fn ncnn_mat_get_c(mat: NcnnMatT) -> c_int;
        pub fn ncnn_mat_get_elemsize(mat: NcnnMatT) -> usize;
        pub fn ncnn_mat_get_elempack(mat: NcnnMatT) -> c_int;
        pub fn ncnn_mat_get_cstep(mat: NcnnMatT) -> usize;
        pub fn ncnn_mat_get_data(mat: NcnnMatT) -> *mut c_void;

        #[allow(dead_code)]
        pub fn ncnn_mat_fill_float(mat: NcnnMatT, value: c_float);
    }
}

struct NcnnRuntimeOption {
    ptr: ffi::NcnnOptionT,
}

impl NcnnRuntimeOption {
    fn new() -> anyhow::Result<Self> {
        let ptr = unsafe { ffi::ncnn_option_create() };
        if ptr.is_null() {
            bail!("Cannot create NCNN runtime options");
        }
        Ok(Self { ptr })
    }

    fn set_num_threads(&mut self, num_threads: u32) {
        unsafe { ffi::ncnn_option_set_num_threads(self.ptr, num_threads as i32) };
    }

    #[cfg(not(feature = "ncnn-cpu"))]
    fn set_use_vulkan_compute(&mut self, enabled: bool) {
        unsafe { ffi::ncnn_option_set_use_vulkan_compute(self.ptr, i32::from(enabled)) };
    }
}

impl Drop for NcnnRuntimeOption {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::ncnn_option_destroy(self.ptr) };
        }
    }
}

struct NcnnNet {
    ptr: ffi::NcnnNetT,
}

impl NcnnNet {
    fn new() -> anyhow::Result<Self> {
        let ptr = unsafe { ffi::ncnn_net_create() };
        if ptr.is_null() {
            bail!("Cannot create NCNN net");
        }
        Ok(Self { ptr })
    }

    fn set_option(&mut self, options: &NcnnRuntimeOption) {
        unsafe { ffi::ncnn_net_set_option(self.ptr, options.ptr) };
    }

    fn load_param(&mut self, path: &Path) -> anyhow::Result<()> {
        let path = path_to_cstring(path)?;
        let status = unsafe { ffi::ncnn_net_load_param(self.ptr, path.as_ptr()) };
        if status != 0 {
            bail!("Error loading NCNN params");
        }
        Ok(())
    }

    fn load_model(&mut self, path: &Path) -> anyhow::Result<()> {
        let path = path_to_cstring(path)?;
        let status = unsafe { ffi::ncnn_net_load_model(self.ptr, path.as_ptr()) };
        if status != 0 {
            bail!("Error loading NCNN weights");
        }
        Ok(())
    }

    fn create_extractor(&self) -> anyhow::Result<NcnnExtractor> {
        let ptr = unsafe { ffi::ncnn_extractor_create(self.ptr) };
        if ptr.is_null() {
            bail!("Cannot create NCNN extractor");
        }
        Ok(NcnnExtractor { ptr })
    }
}

impl Drop for NcnnNet {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::ncnn_net_destroy(self.ptr) };
        }
    }
}

struct NcnnExtractor {
    ptr: ffi::NcnnExtractorT,
}

impl NcnnExtractor {
    fn input(&mut self, name: &str, mat: &NcnnMat) -> anyhow::Result<()> {
        let name = CString::new(name)?;
        let status = unsafe { ffi::ncnn_extractor_input(self.ptr, name.as_ptr(), mat.ptr) };
        if status != 0 {
            bail!("Error setting NCNN input");
        }
        Ok(())
    }

    fn extract(&mut self, name: &str, mat: &mut NcnnMat) -> anyhow::Result<()> {
        let name = CString::new(name)?;
        let status = unsafe { ffi::ncnn_extractor_extract(self.ptr, name.as_ptr(), &mut mat.ptr) };
        if status != 0 {
            bail!("Error extracting NCNN output");
        }
        Ok(())
    }
}

impl Drop for NcnnExtractor {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::ncnn_extractor_destroy(self.ptr) };
        }
    }
}

struct NcnnMat {
    ptr: ffi::NcnnMatT,
}

impl NcnnMat {
    fn new() -> anyhow::Result<Self> {
        let ptr = unsafe { ffi::ncnn_mat_create() };
        if ptr.is_null() {
            bail!("Cannot create NCNN mat");
        }
        Ok(Self { ptr })
    }

    unsafe fn new_external_3d(w: i32, h: i32, c: i32, data: *mut c_void) -> anyhow::Result<Self> {
        let ptr = unsafe { ffi::ncnn_mat_create_external_3d(w, h, c, data, std::ptr::null_mut()) };
        if ptr.is_null() {
            bail!("Cannot create NCNN input mat");
        }
        Ok(Self { ptr })
    }

    fn dims(&self) -> i32 {
        unsafe { ffi::ncnn_mat_get_dims(self.ptr) }
    }

    fn w(&self) -> i32 {
        unsafe { ffi::ncnn_mat_get_w(self.ptr) }
    }

    fn h(&self) -> i32 {
        unsafe { ffi::ncnn_mat_get_h(self.ptr) }
    }

    fn d(&self) -> i32 {
        unsafe { ffi::ncnn_mat_get_d(self.ptr) }
    }

    fn c(&self) -> i32 {
        unsafe { ffi::ncnn_mat_get_c(self.ptr) }
    }

    fn elemsize(&self) -> usize {
        unsafe { ffi::ncnn_mat_get_elemsize(self.ptr) }
    }

    fn elempack(&self) -> i32 {
        unsafe { ffi::ncnn_mat_get_elempack(self.ptr) }
    }

    fn cstep(&self) -> usize {
        unsafe { ffi::ncnn_mat_get_cstep(self.ptr) }
    }

    fn data(&self) -> *mut c_void {
        unsafe { ffi::ncnn_mat_get_data(self.ptr) }
    }
}

impl std::fmt::Debug for NcnnMat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NcnnMat")
            .field("dims", &self.dims())
            .field("c", &self.c())
            .field("d", &self.d())
            .field("h", &self.h())
            .field("w", &self.w())
            .field("elemsize", &self.elemsize())
            .field("elempack", &self.elempack())
            .field("cstep", &self.cstep())
            .finish()
    }
}

impl Drop for NcnnMat {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::ncnn_mat_destroy(self.ptr) };
        }
    }
}

/// Runtime options for the NCNN backend.
#[derive(Debug, Clone)]
pub struct NcnnOptions {
    /// Input blob name. If omitted, the first `Input` layer in the `.param` file is used.
    pub input_name: Option<String>,
    /// Plate-output blob name. If omitted, the first terminal blob in the `.param` file is used.
    pub plate_output_name: Option<String>,
    /// Region-output blob name. If omitted, the second terminal blob is used when the config has
    /// region labels.
    pub region_output_name: Option<String>,
    /// Disable region extraction even when the plate config contains region labels.
    pub disable_region_output: bool,
    /// Number of NCNN worker threads. `None` keeps NCNN's default.
    pub num_threads: Option<u32>,
    /// Enable Vulkan compute when NCNN and the Rust feature are built with Vulkan support.
    pub use_vulkan: bool,
}

impl Default for NcnnOptions {
    fn default() -> Self {
        Self {
            input_name: None,
            plate_output_name: None,
            region_output_name: None,
            disable_region_output: false,
            num_threads: None,
            use_vulkan: false,
        }
    }
}

/// Blob names inferred from an NCNN text param file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NcnnBlobNames {
    pub input_name: String,
    pub output_names: Vec<String>,
}

/// Loaded NCNN model and its blob names.
pub struct NcnnBackend {
    net: NcnnNet,
    input_name: String,
    plate_output_name: String,
    region_output_name: Option<String>,
}

impl NcnnBackend {
    pub fn from_files(
        param_path: impl AsRef<Path>,
        bin_path: impl AsRef<Path>,
        config: &PlateConfig,
        options: NcnnOptions,
    ) -> anyhow::Result<Self> {
        let param_path = param_path.as_ref();
        let bin_path = bin_path.as_ref();

        if !param_path.exists() {
            bail!("NCNN param file not found: {}", param_path.display());
        }
        if !bin_path.exists() {
            bail!("NCNN bin file not found: {}", bin_path.display());
        }

        let inferred = infer_blob_names(param_path).with_context(|| {
            format!(
                "Cannot infer NCNN blob names from {}. Pass explicit blob names instead.",
                param_path.display()
            )
        })?;

        let input_name = options
            .input_name
            .unwrap_or_else(|| inferred.input_name.clone());
        let plate_output_name = options
            .plate_output_name
            .or_else(|| inferred.output_names.first().cloned())
            .unwrap_or_else(|| "out0".to_owned());
        let region_output_name =
            if config.has_region_recognition() && !options.disable_region_output {
                options
                    .region_output_name
                    .or_else(|| inferred.output_names.get(1).cloned())
                    .or_else(|| Some("out1".to_owned()))
            } else {
                None
            };

        let mut runtime_options = NcnnRuntimeOption::new()?;
        if let Some(num_threads) = options.num_threads {
            runtime_options.set_num_threads(num_threads);
        }
        if options.use_vulkan {
            #[cfg(feature = "ncnn-cpu")]
            eprintln!(
                "Warning: --vulkan was requested, but this binary was built with the ncnn-cpu feature. Vulkan is disabled."
            );
            #[cfg(not(feature = "ncnn-cpu"))]
            runtime_options.set_use_vulkan_compute(true);
        }

        let mut net = NcnnNet::new()?;
        net.set_option(&runtime_options);
        net.load_param(param_path)
            .with_context(|| format!("Cannot load NCNN params: {}", param_path.display()))?;
        net.load_model(bin_path)
            .with_context(|| format!("Cannot load NCNN weights: {}", bin_path.display()))?;

        Ok(Self {
            net,
            input_name,
            plate_output_name,
            region_output_name,
        })
    }

    pub fn has_region_output(&self) -> bool {
        self.region_output_name.is_some()
    }

    pub fn run_raw(
        &self,
        raw_nhwc: &[u8],
        config: &PlateConfig,
        with_region: bool,
    ) -> anyhow::Result<(Vec<f32>, Option<Vec<f32>>)> {
        let h = config.img_height as usize;
        let w = config.img_width as usize;
        let c = config.num_channels() as usize;
        let expected = h * w * c;
        if raw_nhwc.len() != expected {
            bail!(
                "Unexpected NCNN input length: got {}, expected {}",
                raw_nhwc.len(),
                expected
            );
        }

        let mut input_data: Vec<f32> = raw_nhwc.iter().map(|&v| v as f32).collect();
        let input = unsafe {
            // pnnx preserves the ONNX input tensor order. The existing fast-plate-ocr models use
            // NHWC shape [1, H, W, C], which maps to NCNN Mat dimensions w=C, h=W, c=H.
            NcnnMat::new_external_3d(
                c as i32,
                w as i32,
                h as i32,
                input_data.as_mut_ptr().cast::<c_void>(),
            )?
        };

        let mut extractor = self.net.create_extractor()?;
        extractor
            .input(&self.input_name, &input)
            .with_context(|| format!("Cannot set NCNN input blob '{}'", self.input_name))?;

        let mut plate_output = NcnnMat::new()?;
        extractor
            .extract(&self.plate_output_name, &mut plate_output)
            .with_context(|| {
                format!(
                    "Cannot extract NCNN plate output blob '{}'",
                    self.plate_output_name
                )
            })?;
        let plate_data = mat_to_vec_f32(&plate_output)
            .with_context(|| format!("Cannot read NCNN output {:?}", plate_output))?;

        let region_data = if with_region {
            if let Some(name) = &self.region_output_name {
                let mut region_output = NcnnMat::new()?;
                extractor
                    .extract(name, &mut region_output)
                    .with_context(|| format!("Cannot extract NCNN region output blob '{name}'"))?;
                Some(
                    mat_to_vec_f32(&region_output)
                        .with_context(|| format!("Cannot read NCNN output {:?}", region_output))?,
                )
            } else {
                None
            }
        } else {
            None
        };

        Ok((plate_data, region_data))
    }
}

fn path_to_cstring(path: &Path) -> anyhow::Result<CString> {
    let path = path
        .to_str()
        .with_context(|| format!("Path is not valid UTF-8: {}", path.display()))?;
    Ok(CString::new(path)?)
}

fn mat_to_vec_f32(mat: &NcnnMat) -> anyhow::Result<Vec<f32>> {
    if mat.elemsize() != std::mem::size_of::<f32>() {
        bail!(
            "NCNN output uses {}-byte elements; only f32 outputs are supported",
            mat.elemsize()
        );
    }
    if mat.elempack() != 1 {
        bail!(
            "NCNN output uses elempack={}; packed outputs are not supported",
            mat.elempack()
        );
    }

    let dims = mat.dims();
    let w = mat.w().max(1) as usize;
    let h = mat.h().max(1) as usize;
    let d = mat.d().max(1) as usize;
    let c = mat.c().max(1) as usize;
    let plane_len = match dims {
        1 => w,
        2 => w * h,
        3 => w * h,
        4 => w * h * d,
        _ => bail!("Unsupported NCNN output dimensions: {dims}"),
    };
    let channels = if dims >= 3 { c } else { 1 };
    let cstep = if dims >= 3 { mat.cstep() } else { plane_len };
    let data = mat.data().cast::<f32>();
    if data.is_null() {
        bail!("NCNN output data pointer is null");
    }

    let mut out = Vec::with_capacity(plane_len * channels);
    unsafe {
        for ch in 0..channels {
            let start = data.add(ch * cstep);
            out.extend_from_slice(std::slice::from_raw_parts(start, plane_len));
        }
    }
    Ok(out)
}

pub fn infer_blob_names(param_path: &Path) -> anyhow::Result<NcnnBlobNames> {
    let text = std::fs::read_to_string(param_path)
        .with_context(|| format!("Cannot read NCNN param file: {}", param_path.display()))?;
    infer_blob_names_from_str(&text)
}

pub(crate) fn infer_blob_names_from_str(text: &str) -> anyhow::Result<NcnnBlobNames> {
    let mut input_name = None;
    let mut produced = Vec::new();
    let mut consumed = HashSet::new();

    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        if line == "7767517" {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 4 {
            continue;
        }

        let bottom_count = match tokens[2].parse::<usize>() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let top_count = match tokens[3].parse::<usize>() {
            Ok(n) => n,
            Err(_) => continue,
        };

        let names_start = 4;
        let bottoms_end = names_start + bottom_count;
        let tops_end = bottoms_end + top_count;
        if tokens.len() < tops_end {
            continue;
        }

        for bottom in &tokens[names_start..bottoms_end] {
            consumed.insert((*bottom).to_owned());
        }

        let tops = &tokens[bottoms_end..tops_end];
        if tokens[0] == "Input" && input_name.is_none() {
            input_name = tops.first().map(|s| (*s).to_owned());
        }
        for top in tops {
            produced.push((*top).to_owned());
        }
    }

    let input_name = input_name.unwrap_or_else(|| "in0".to_owned());
    let output_names = produced
        .into_iter()
        .filter(|name| !consumed.contains(name))
        .collect::<Vec<_>>();

    Ok(NcnnBlobNames {
        input_name,
        output_names,
    })
}

pub fn ncnn_paths_for_onnx(onnx_path: &Path) -> anyhow::Result<(PathBuf, PathBuf)> {
    let parent = onnx_path.parent().unwrap_or_else(|| Path::new(""));
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
    use super::infer_blob_names_from_str;

    #[test]
    fn infers_input_and_terminal_outputs() {
        let param = r#"
7767517
4 4
Input in0 0 1 in0 0=1 1=140 2=70
Convolution conv 1 1 in0 hidden 0=32
Softmax plate 1 1 hidden out0 0=1
Softmax region 1 1 hidden out1 0=1
"#;

        let names = infer_blob_names_from_str(param).unwrap();
        assert_eq!(names.input_name, "in0");
        assert_eq!(names.output_names, vec!["out0", "out1"]);
    }
}
