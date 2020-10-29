# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "analysis",
    srcs = [
        "affine_expr.cc",
        "affine_constraints.cc",
        "strides.cc",
        "uses.cc",
    ],
    hdrs = [
        "affine_expr.h",
        "affine_constraints.h",
        "strides.h",
        "uses.h",
    ],
    deps = [
        "//pmlc/dialect/pxa/ir",
        "//pmlc/dialect/stdx/ir",
        "//pmlc/util/bilp",
        "//pmlc/util/math",
    ],
)
