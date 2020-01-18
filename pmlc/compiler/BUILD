# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "compiler",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    copts = [
    	"-Iexternal/vulkan_sdk_include",
    ],
    deps = [
        "//base/util",
        "//pmlc/conversion/tile_to_pxa",
        "//pmlc/tools/pmlc-vulkan-runner:MLIRVulkanRuntime",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:SPIRVDialect",
		"@llvm-project//mlir:SPIRVLowering", 
		"@llvm-project//mlir:SPIRVSerialization", 
    ],
)
