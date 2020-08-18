# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "intel_gen",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//pmlc/compiler",
        "//pmlc/conversion/SCFToGPU",
        "//pmlc/conversion/gpu",
        "//pmlc/conversion/pxa_to_affine",
        "//pmlc/conversion/tile_to_pxa",
        "//pmlc/dialect/pxa/transforms",
        "//pmlc/dialect/tile/transforms",
        "//pmlc/dialect/stdx/transforms",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:GPUToSPIRVTransforms",
        "@llvm-project//mlir:GPUToVulkanTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:SCFToGPUPass",
    ],
    alwayslink = 1,
)
