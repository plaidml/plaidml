package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

exports_files([
    "plaidml.yml",
    "sphinx.yml",
])

pkg_tar(
    name = "pkg",
    srcs = glob(["*.yml"]),
    package_dir = "conda",
    strip_prefix = ".",
    tags = [
        "deb",
        "git_big",
    ],
)
