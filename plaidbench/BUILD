# Copyright 2018 Intel Corporation.

package(default_visibility = ["//visibility:public"])

exports_files([
    "requirements.txt",
])

load("@rules_python//python:defs.bzl", "py_binary")
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

py_library(
    name = "plaidbench",
    srcs = ["plaidbench.py"],
    deps = [":py"],
)

py_binary(
    name = "bin",
    srcs = ["plaidbench.py"],
    main = "plaidbench.py",
    deps = [
        ":py",
        "//plaidml/keras",
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
