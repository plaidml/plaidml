# Copyright 2020 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "transforms",
    srcs = [
        "contraction.cc",
    ],
    hdrs = [
        "contraction.h",
    ],
    deps = [
        "//pmlc/dialect/tile/ir",
        "//pmlc/util",
        "//pmlc/util/bilp",
        "//pmlc/util/math",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)
