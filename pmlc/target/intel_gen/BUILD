# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "intel_gen",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//pmlc/compiler",
        "//pmlc/conversion/pxa_to_affine",
        "@llvm-project//mlir:AffineDialectRegistration",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:GPUDialectRegistration",
        "@llvm-project//mlir:GPUToSPIRVTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LoopDialectRegistration",
        "@llvm-project//mlir:LoopsToGPUPass",
        "@llvm-project//mlir:SPIRVDialectRegistration",
        "@llvm-project//mlir:StandardDialectRegistration",
    ],
    alwayslink = 1,
)
