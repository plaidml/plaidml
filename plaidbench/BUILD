# Copyright 2018 Intel Corporation.

package(default_visibility = ["//visibility:public"])

exports_files([
    "requirements.txt",
])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("//bzl:conda.bzl", "conda_binary")
load(
    "//bzl:plaidml.bzl",
    "plaidml_py_library",
    "plaidml_py_wheel",
)

filegroup(
    name = "golden",
    srcs = glob(["plaidbench/golden/**"]),
)

filegroup(
    name = "networks",
    srcs = glob(["plaidbench/networks/**"]),
)

filegroup(
    name = "cifar16",
    srcs = ["plaidbench/cifar16.npy"],
)

pkg_tar(
    name = "pkg",
    srcs = glob(["**/*"]),
    package_dir = "plaidbench",
    strip_prefix = ".",
)

py_library(
    name = "plaidbench",
    srcs = ["plaidbench.py"],
    deps = [":py"],
)

conda_binary(
    name = "bin",
    srcs = ["plaidbench.py"],
    env = "@com_intel_plaidml_conda//:env",
    main = "plaidbench.py",
    deps = [
        ":py",
        "//plaidml/keras",
        "//plaidml2/bridge/keras",
    ],
)

plaidml_py_library(
    name = "py",
    srcs = [
        "plaidbench/__init__.py",
        "plaidbench/cli.py",
        "plaidbench/core.py",
        "plaidbench/frontend_keras.py",
        "plaidbench/networks/__init__.py",
        "plaidbench/networks/keras/__init__.py",
        "plaidbench/networks/keras/imdb_lstm.py",
        "plaidbench/networks/keras/inception_v3.py",
        "plaidbench/networks/keras/mobilenet.py",
        "plaidbench/networks/keras/mobilenet_v2.py",
        "plaidbench/networks/keras/resnet50.py",
        "plaidbench/networks/keras/vgg16.py",
        "plaidbench/networks/keras/vgg19.py",
        "plaidbench/networks/keras/xception.py",
    ],
    data = glob([
        "plaidbench/cifar16.npy",
        "plaidbench/imdb16.npy",
        "plaidbench/golden/**",
        "plaidbench/networks/**",
    ]),
)

plaidml_py_wheel(
    name = "wheel",
    package_name = "plaidbench",
    srcs = [":py"],
    config = "not_setup.cfg",
    python = "py2.py3",
)
