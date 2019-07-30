# Copyright 2018 Intel Corporation.

load("//bzl:conda.bzl", "conda_binary")
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

conda_binary(
    name = "harness",
    srcs = ["harness.py"],
    data = glob(["examples/**"]),
    env = "@com_intel_plaidml_conda_ml_gpu//:env",
    deps = [
        "//plaidml/keras",
    ],
)

conda_binary(
    name = "tune",
    srcs = ["tune.py"],
    data = glob(["examples/**"]),
    env = "@com_intel_plaidml_conda_ml_gpu//:env",
    deps = [
        "//plaidml/keras",
    ],
)

pkg_tar(
    name = "pkg",
    srcs = glob(
        ["**/*"],
        exclude = ["__pycache__/*"],
    ),
    package_dir = "networks/keras",
    strip_prefix = ".",
    tags = [
        "deb",
    ],
)
