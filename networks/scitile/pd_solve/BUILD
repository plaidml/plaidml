# Copyright 2018 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "gradtest",
    srcs = [
        "gradtest.py",
    ],
    deps = [
        "//plaidml2:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
    ],
)


