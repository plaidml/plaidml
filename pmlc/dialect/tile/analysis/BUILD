# Copyright 2020 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "analysis",
    srcs = [],
    hdrs = [
        "conv2d_finder.h",
    ],
    deps = [
        "//pmlc/dialect/tile/ir",
        "@llvm-project//mlir:IR",
    ],
)
