# Copyright 2018 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_binary","plaidml_cc_test")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "reorgyolo",
    srcs = ["reorgyolo.py"],
    deps = [
        "//plaidml2:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
        "//plaidml2/op:py",
    ],
)