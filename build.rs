use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-env-changed=NCNN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=NCNN_LINK_KIND");
    println!("cargo:rerun-if-env-changed=NCNN_LIB_NAME");
    println!("cargo:rerun-if-env-changed=NCNN_CXX_STDLIB");
    println!("cargo:rerun-if-env-changed=NCNN_OPENMP_LIB");
    println!("cargo:rerun-if-env-changed=NCNN_VULKAN");
    println!("cargo:rerun-if-env-changed=NCNN_VULKAN_LIBS");
    println!("cargo:rerun-if-env-changed=NCNN_EXTRA_LIBS");

    if env::var_os("CARGO_FEATURE_NCNN").is_none() {
        return;
    }

    if let Some(lib_dir) = env::var_os("NCNN_LIB_DIR") {
        println!(
            "cargo:rustc-link-search=native={}",
            lib_dir.to_string_lossy()
        );
    }

    let link_kind = env::var("NCNN_LINK_KIND").unwrap_or_else(|_| "static".to_owned());
    let lib_name = env::var("NCNN_LIB_NAME").unwrap_or_else(|_| "ncnn".to_owned());
    println!("cargo:rustc-link-lib={link_kind}={lib_name}");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        let cxx = env::var("NCNN_CXX_STDLIB").unwrap_or_else(|_| "stdc++".to_owned());
        println!("cargo:rustc-link-lib=dylib={cxx}");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=m");

        if link_kind == "static" {
            let openmp = env::var("NCNN_OPENMP_LIB").unwrap_or_else(|_| "gomp".to_owned());
            for lib in split_link_libs(&openmp).filter(|lib| !is_disabled(lib)) {
                println!("cargo:rustc-link-lib=dylib={lib}");
            }
        }

        if env_flag("NCNN_VULKAN") || env::var_os("CARGO_FEATURE_NCNN_VULKAN").is_some() {
            println!("cargo:rustc-link-lib=dylib=vulkan");

            if link_kind == "static" {
                link_static_vulkan_shader_libs();
            }
        }
    } else if target_os == "macos" {
        let cxx = env::var("NCNN_CXX_STDLIB").unwrap_or_else(|_| "c++".to_owned());
        println!("cargo:rustc-link-lib=dylib={cxx}");
    }

    if let Ok(extra_libs) = env::var("NCNN_EXTRA_LIBS") {
        for lib in split_link_libs(&extra_libs).filter(|lib| !is_disabled(lib)) {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}

fn link_static_vulkan_shader_libs() {
    if let Ok(vulkan_libs) = env::var("NCNN_VULKAN_LIBS") {
        link_libs(&vulkan_libs);
        return;
    }

    if link_pkg_config_libs(&["glslang", "spirv"]) {
        return;
    }

    link_libs("glslang,MachineIndependent,GenericCodeGen,SPIRV,OSDependent");
}

fn link_pkg_config_libs(packages: &[&str]) -> bool {
    let output = Command::new("pkg-config")
        .arg("--libs")
        .arg("--static")
        .args(packages)
        .output();

    let Ok(output) = output else {
        return false;
    };
    if !output.status.success() {
        return false;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut linked_any = false;
    for token in stdout.split_whitespace() {
        if let Some(path) = token.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        } else if let Some(lib) = token.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
            linked_any = true;
        } else if token == "-pthread" {
            println!("cargo:rustc-link-lib=pthread");
        }
    }

    linked_any
}

fn link_libs(value: &str) {
    for lib in split_link_libs(value).filter(|lib| !is_disabled(lib)) {
        println!("cargo:rustc-link-lib={lib}");
    }
}

fn split_link_libs(value: &str) -> impl Iterator<Item = &str> {
    value
        .split(|c| c == ',' || c == ';' || c == ' ')
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn is_disabled(value: &str) -> bool {
    matches!(
        value,
        "0" | "false" | "False" | "off" | "OFF" | "none" | "NONE"
    )
}

fn env_flag(name: &str) -> bool {
    env::var(name).map_or(false, |value| {
        let value = value.trim();
        !value.is_empty() && !is_disabled(value)
    })
}
