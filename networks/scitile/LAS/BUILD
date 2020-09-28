# Copyright 2018 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "conjgrad",
    srcs = [
        "conjgrad.py",
    ],
    deps = [
        "//plaidml:py",
    ],
    data = glob(["test_data/*.npz"]),
    args =["20x20"]
)

py_binary(
    name = "solver_timings",
    srcs = [
        "conjgrad.py",
        "solver_timings.py",
    ],
    deps = [
        "//plaidml:py",
    ],
    data = glob(["test_data/*.npz"])
)

py_test(
    name = "cg_tests",
    srcs = [
        "conjgrad.py",
        "cg_tests.py"
    ],
    deps = [
        "//plaidml:py",
    ],
    data = glob(["test_data/*.npz"])
)

