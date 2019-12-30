# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "intel_gen",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//pmlc/compiler",
        "//pmlc/dialect/pxa",
        "@mlir//:AffineDialectRegistration",
        "@mlir//:AffineToStandardTransforms",
        "@mlir//:GPUDialectRegistration",
        "@mlir//:GPUToSPIRVTransforms",
        "@mlir//:GPUTransforms",
        "@mlir//:LLVMTransforms",
        "@mlir//:LoopDialectRegistration",
        "@mlir//:LoopsToGPUPass",
        "@mlir//:SPIRVDialectRegistration",
        "@mlir//:StandardDialectRegistration",
    ],
    alwayslink = 1,
)
