# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "tile",
    srcs = [
        "builder.cc",
        "gradient.cc",
        "gradient.h",
    ],
    hdrs = [
        "builder.h",
        "program.h",
    ],
    deps = [
        "//pmlc/dialect/eltwise",
        "//pmlc/dialect/tile/ir",
        "//pmlc/dialect/tile/transforms",
        "//pmlc/util",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
    ],
)
