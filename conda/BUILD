package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

exports_files([
    "buildkite.yml",
    "ml_gpu.yml",
    "plaidml.yml",
    "pytorch.yml",
    "sphinx.yml",
    "tools-unix.yml",
    "tools-windows.yml",
])

pkg_tar(
    name = "pkg",
    srcs = glob(["*.yml"]),
    package_dir = "conda",
    strip_prefix = ".",
    tags = [
        "deb",
    ],
)
