# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "tile",
    srcs = [
        "builder.cc",
        "gradient.cc",
        "gradient.h",
    ],
    hdrs = [
        "builder.h",
    ],
    deps = [
        "//pmlc/compiler",
        "//pmlc/dialect/eltwise",
        "//pmlc/dialect/tile/ir",
        "//pmlc/dialect/tile/transforms",
        "//pmlc/util",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)
