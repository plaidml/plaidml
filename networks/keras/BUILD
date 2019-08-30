# Copyright 2018 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "harness",
    srcs = ["harness.py"],
    data = glob(["examples/**"]),
    deps = [
        "//plaidml/keras",
    ],
)

py_binary(
    name = "tune",
    srcs = ["tune.py"],
    data = glob(["examples/**"]),
    deps = [
        "//plaidml/keras",
    ],
)
