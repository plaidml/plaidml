# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:mlir.bzl", "COPTS")

plaidml_cc_library(
    name = "stripe_to_affine",
    srcs = ["convert_stripe_to_affine.cc"],
    hdrs = ["convert_stripe_to_affine.h"],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        "//pmlc/dialect/stripe",
    ],
    alwayslink=1,
)
