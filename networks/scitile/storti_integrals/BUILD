# Copyright 2018 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "torus",
    srcs = ["torus.py"],
    deps = [
        ":op",
        "//plaidml:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
)

py_library(
    name = "op",
    srcs = ["op.py"],
    deps = [
        "//plaidml:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
)

py_binary(
    name = "hypersphere",
    srcs = ["hypersphere.py"],
    deps = [
        ":op",
        "//plaidml:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
)

py_test(
    name = "storti_integral_tests",
    srcs = [
        "hypersphere.py",
        "storti_integral_tests.py",
        "torus.py",
    ],
    deps = [
        ":op",
        "//plaidml:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
)
