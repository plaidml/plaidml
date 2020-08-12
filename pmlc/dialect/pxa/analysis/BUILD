# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "analysis",
    srcs = [
        "affine_expr.cc",
        "strides.cc",
        "uses.cc",
    ],
    hdrs = [
        "affine_expr.h",
        "strides.h",
        "uses.h",
    ],
    deps = [
        "//pmlc/dialect/pxa/ir",
    ],
)
