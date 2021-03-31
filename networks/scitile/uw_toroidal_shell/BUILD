# Copyright 2018 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_binary", "py_test")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "uw_toroidal_shell",
    srcs = ["uw_toroidal_shell.py"],
    deps = [
        "//plaidml:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
)

py_test(
    name = "uw_tests",
    size = "medium",
    srcs = [
        "uw_tests.py",
        "uw_toroidal_shell.py",
    ],
    args = ["-v"],
    deps = [
        "//plaidml:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
)
