# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:mlir.bzl", "COPTS")

plaidml_cc_library(
    name = "affine_to_stripe",
    srcs = [
        "convert_affine_to_stripe.cc",
        "convert_affine_to_stripe.h",
        "supported_ops.inc"
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        "//pmlc/dialect/stripe",
    ],
    alwayslink=1,
)
