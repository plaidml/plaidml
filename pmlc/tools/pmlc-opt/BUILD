# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

plaidml_cc_library(
    name = "lib",
    srcs = ["pmlc-opt.cc"],
    deps = [
        "//pmlc/conversion/pxa_to_affine",
        "//pmlc/conversion/stdx_to_llvm",
        "//pmlc/conversion/tile_to_pxa",
        "//pmlc/dialect/eltwise",
        "//pmlc/dialect/pxa",
        "//pmlc/dialect/stdx",
        "//pmlc/dialect/tile",
        "//pmlc/dialect/xsmm",
        "//pmlc/target/x86",
        "//pmlc/util",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:LoopsToGPUPass",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

plaidml_cc_binary(
    name = "pmlc-opt",
    deps = [":lib"],
)
