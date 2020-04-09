# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_test")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "reorgyolo",
    srcs = ["reorgyolo.py"],
    deps = [
        "//plaidml:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
        "//plaidml/op:py",
    ],
)
