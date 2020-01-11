# Copyright 2018 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("@rules_python//python:defs.bzl", "py_binary")

exports_files([
    "requirements.txt",
])

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
        "//plaidml2/bridge/keras",
    ],
)

py_library(
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

# TODO: use py_setup
# plaidml_py_wheel(
#     name = "wheel",
#     package_name = "plaidbench",
#     srcs = [":py"],
#     config = "not_setup.cfg",
#     python = "py2.py3",
# )
