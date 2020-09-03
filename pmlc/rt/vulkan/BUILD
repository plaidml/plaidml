# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "vulkan",
    srcs = [
        "vulkan_runtime.cc",
        "wrappers.cc",
    ],
    hdrs = [
        "vulkan_runtime.h",
    ],
    local_defines = ["VULKAN_RT_BUILD"],
    deps = [
        "//pmlc/compiler",
        "@half",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVLowering",
        "@llvm-project//mlir:mlir_runner_utils",
        "@volk",
    ],
    alwayslink = 1,
)

plaidml_cc_binary(
    name = "vkprobe",
    srcs = ["vkprobe.cc"],
    deps = ["@volk"],
)
