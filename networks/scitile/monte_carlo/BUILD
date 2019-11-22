# Copyright 2018 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "pi",
    srcs = ["pi.py"],
    deps = [
        "//plaidml2:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
    ],
)

py_test(
    name = "pi_tests",
    srcs = [
        "pi_tests.py",
        "pi.py",
    ],
    deps = [
        "//plaidml2:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
    ],
)

