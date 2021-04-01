# Copyright 2018 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "min_ex",
    srcs = [
        "min_ex.py",
    ],
    deps = [
        "//plaidml:py",
        "//plaidml/op:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
)

py_binary(
    name = "conjgrad",
    srcs = [
        "conjgrad.py",
    ],
    deps = [
        "//plaidml:py",
        "//plaidml/op:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
    data = glob(["test_data/*.npz"]),
    args =["10x10"]
)

py_test(
    name = "cg_tests",
    srcs = [
        "conjgrad.py",
        "cg_tests.py"
    ],
    deps = [
        "//plaidml:py",
        "//plaidml/op:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
    data = glob(["test_data/*.npz"])
)

