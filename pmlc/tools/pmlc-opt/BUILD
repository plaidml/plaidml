# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary")

plaidml_cc_binary(
    name = "pmlc-opt",
    srcs = ["pmlc-opt.cc"],
    deps = [
        "//pmlc/conversion/pxa_to_affine",
        "//pmlc/conversion/tile_to_pxa",
        "//pmlc/dialect/eltwise",
        "//pmlc/dialect/pxa",
        "//pmlc/dialect/stdx",
        "//pmlc/dialect/tile",
        "//pmlc/target/x86",
        "@llvm-project//mlir:AffineDialectRegistration",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:StandardDialectRegistration",
        "@llvm-project//mlir:Transforms",
    ],
)
