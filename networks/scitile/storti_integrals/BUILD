# Copyright 2018 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "torus",
    srcs = ["torus.py",],
    deps = [
        ":op",
        "//plaidml2:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
    ],
)

py_library(
    name="op",
    srcs = ["op.py"],
    deps =[
        "//plaidml2:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
    ],
)

py_binary(
    name = "hypersphere",
    srcs = ["hypersphere.py",],
    deps = [
        ":op",
        "//plaidml2:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
    ],
)

py_test(
    name = "storti_integral_tests",
    srcs = [
        "storti_integral_tests.py",
        "torus.py",
        "hypersphere.py",
    ],
    deps = [
        ":op",
        "//plaidml2:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
    ],
)

