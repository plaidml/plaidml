# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary")

plaidml_cc_binary(
    name = "pmlc-opt",
    srcs = ["pmlc-opt.cc"],
    deps = [
        "//pmlc/conversion/affine_to_stripe",
        "//pmlc/conversion/pxa_to_affine",
        "//pmlc/conversion/stripe_to_affine",
        "//pmlc/conversion/tile_to_pxa",
        "//pmlc/conversion/tile_to_stripe",
        "//pmlc/dialect/eltwise",
        "//pmlc/dialect/pxa",
        "//pmlc/dialect/stripe",
        "//pmlc/dialect/stripe:passes",
        "//pmlc/dialect/tile",
        "@llvm-project//mlir:AffineDialectRegistration",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:StandardDialectRegistration",
        "@llvm-project//mlir:Transforms",
    ],
)
