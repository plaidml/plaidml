# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "intel_gen",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//pmlc/compiler",
        "//pmlc/conversion/gpu_to_vulkan",
        "//pmlc/conversion/pxa_to_affine",
        "//pmlc/conversion/tile_to_pxa",
        "//pmlc/dialect/tile/transforms",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:GPUToSPIRVTransforms",
        "@llvm-project//mlir:GPUToVulkanTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LoopsToGPUPass",
    ],
    alwayslink = 1,
)
