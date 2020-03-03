# Copyright 2020 Intel Corporation

load("//vendor/bazel:repo.bzl", "http_archive")
load("//vendor/conda:repo.bzl", "conda_repo")
load("//vendor/openvino:repo.bzl", "openvino_workspace")
load("//vendor/xsmm:repo.bzl", "xsmm_repo")

# Sanitize a dependency so that it works correctly from code that includes it as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def plaidml_workspace():
    configure_toolchain()

    http_archive(
        name = "bazel_latex",
        sha256 = "5119802a5fbe2f27914af455c59b4ecdaaf57c0bc6c63da38098a30d94f48c9a",
        strip_prefix = "bazel-latex-b6375d9df2952548c3371c0c865710655e8b1cc1",
        url = "https://github.com/plaidml/bazel-latex/archive/b6375d9df2952548c3371c0c865710655e8b1cc1.zip",
    )

    http_archive(
        name = "boost",
        url = "https://github.com/plaidml/depot/raw/master/boost_1_66_0.tar.gz",
        sha256 = "bd0df411efd9a585e5a2212275f8762079fed8842264954675a4fddc46cfcf60",
        strip_prefix = "boost_1_66_0",
        build_file = clean_dep("//bzl:boost.BUILD"),
    )

    http_archive(
        name = "com_github_google_benchmark",
        url = "https://github.com/google/benchmark/archive/v1.5.0.tar.gz",
        sha256 = "3c6a165b6ecc948967a1ead710d4a181d7b0fbcaa183ef7ea84604994966221a",
        strip_prefix = "benchmark-1.5.0",
    )

    http_archive(
        name = "easylogging",
        url = "https://github.com/muflihun/easyloggingpp/releases/download/v9.95.0/easyloggingpp_v9.95.0.tar.gz",
        sha256 = "4b1aebe19e383349c6e438aac357eccfabb0ce34430e872508ed8ee0d1629e0f",
        build_file = clean_dep("//bzl:easylogging.BUILD"),
    )

    http_archive(
        name = "gflags",
        url = "https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
        sha256 = "7d17922978692175c67ef5786a014df44bfbfe3b48b30937cca1413d4ff65f75",
        strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    )

    http_archive(
        name = "gmock",
        url = "https://github.com/google/googletest/archive/release-1.8.0.tar.gz",
        sha256 = "58a6f4277ca2bc8565222b3bbd58a177609e9c488e8a72649359ba51450db7d8",
        strip_prefix = "googletest-release-1.8.0",
        build_file = clean_dep("//bzl:gmock.BUILD"),
    )

    http_archive(
        name = "io_bazel_rules_jsonnet",
        sha256 = "d05d719c4738e8aac5f13b32f745ff4832b9638ecc89ddcb6e36c379a1ada025",
        strip_prefix = "rules_jsonnet-0.1.0",
        url = "https://github.com/bazelbuild/rules_jsonnet/archive/0.1.0.zip",
    )

    http_archive(
        name = "jsonnet",
        url = "https://github.com/google/jsonnet/archive/v0.13.0.zip",
        sha256 = "e9f7095dd2a383001188aa622edaf82059732e11d74f8d0bfdfa84f2682dd547",
        strip_prefix = "jsonnet-0.13.0",
    )

    http_archive(
        name = "pybind11",
        url = "https://github.com/pybind/pybind11/archive/v2.2.4.tar.gz",
        sha256 = "b69e83658513215b8d1443544d0549b7d231b9f201f6fc787a2b2218b408181e",
        strip_prefix = "pybind11-2.2.4",
        build_file = clean_dep("//bzl:pybind11.BUILD"),
    )

    http_archive(
        name = "rules_pkg",
        sha256 = "e46b4f5aa71d1037c7c8142e2fedb503127af4bbd9dbde4a742d119749f68a3f",
        strip_prefix = "rules_pkg-cb54c427343aa48c32e3c09ddcc8f6316cdbd5a6/pkg",
        url = "https://github.com/bazelbuild/rules_pkg/archive/cb54c427343aa48c32e3c09ddcc8f6316cdbd5a6.tar.gz",
    )

    http_archive(
        name = "rules_python",
        sha256 = "b5bab4c47e863e0fbb77df4a40c45ca85f98f5a2826939181585644c9f31b97b",
        strip_prefix = "rules_python-9d68f24659e8ce8b736590ba1e4418af06ec2552",
        url = "https://github.com/bazelbuild/rules_python/archive/9d68f24659e8ce8b736590ba1e4418af06ec2552.tar.gz",
    )

    http_archive(
        name = "zlib",
        url = "https://github.com/plaidml/depot/raw/master/zlib-1.2.8.tar.gz",
        sha256 = "36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d",
        build_file = clean_dep("//bzl:zlib.BUILD"),
    )

    conda_repo(
        name = "com_intel_plaidml_conda_unix",
        env = clean_dep("//conda:unix.yml"),
        build_file = clean_dep("//conda:unix.BUILD"),
    )

    conda_repo(
        name = "com_intel_plaidml_conda_windows",
        env = clean_dep("//conda:windows.yml"),
        build_file = clean_dep("//conda:windows.BUILD"),
    )

    xsmm_repo(
        name = "xsmm",
        url = "https://github.com/hfp/libxsmm/archive/1.12.1.zip",
        sha256 = "451ec9d30f0890bf3081aa3d0d264942a6dea8f9d29c17bececc8465a10a832b",
        strip_prefix = "libxsmm-1.12.1",
        build_file = clean_dep("//vendor/xsmm:xsmm.BUILD"),
    )

    http_archive(
        name = "tbb",
        url = "https://github.com/intel/tbb/archive/v2020.1.zip",
        sha256 = "b81f5dcd7614b7fde305d540d598e3abede9379402615c9514daf09c484333de",
        strip_prefix = "tbb-2020.1",
        build_file = clean_dep("//vendor/tbb:tbb.BUILD"),
    )


    http_archive(
        name = "vulkan_headers",
        url = "https://github.com/KhronosGroup/Vulkan-Headers/archive/v1.2.132.zip",
        sha256 = "e6b5418e3d696ffc7c97991094ece7cafc4c279c8a88029cc60e587bc0c26068",
        strip_prefix = "Vulkan-Headers-1.2.132",
        build_file = clean_dep("//bzl:vulkan_headers.BUILD"),
    )

    http_archive(
        name = "vulkan_loader",
        url = "https://github.com/KhronosGroup/Vulkan-Loader/archive/v1.2.132.zip",
        sha256 = "f42c10bdfaf2ec29d1e4276bf115387852a1dc6aee940f25aff804cc0138d10a",
        strip_prefix = "Vulkan-Loader-1.2.132",
        build_file = clean_dep("//vendor/vulkan_loader:vulkan_loader.BUILD"),
        patches = [clean_dep("//vendor/vulkan_loader:vulkan_loader.patch")],
    )

    LLVM_COMMIT = "9fff6e823cf79075d1f386e1e875b73405368620"
    LLVM_SHA256 = "ff6cfa15aba95405f3ee47580daf34aafebbf07306518e6ab3dced9b18437523"
    LLVM_URL = "https://github.com/plaidml/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)
    http_archive(
        name = "llvm-project",
        url = LLVM_URL,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        link_files = {
            clean_dep("//vendor/llvm:llvm.BUILD"): "llvm/BUILD.bazel",
            clean_dep("//vendor/mlir:mlir.BUILD"): "mlir/BUILD.bazel",
            clean_dep("//vendor/mlir:test.BUILD"): "mlir/test/BUILD.bazel",
        },
        override = "PLAIDML_LLVM_REPO",
    )

    openvino_workspace()

def configure_toolchain():
    http_archive(
        name = "crosstool_ng_linux_x86_64_gcc_8.3.0",
        build_file = clean_dep("//toolchain:crosstool_ng/linux_x86_64.BUILD"),
        sha256 = "091f5732882a499c6b9fb5fcb895176d0c96e958236e16b61d1a9cafec4271ad",
        strip_prefix = "x86_64-unknown-linux-gnu",
        url = "https://github.com/plaidml/depot/raw/master/toolchain/gcc-8.3/x86_64-unknown-linux-gnu-20191010.tgz",
    )
