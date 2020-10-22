# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "vulkan",
    srcs = [
        "vulkan_device.cc",
        "vulkan_device.h",
        "vulkan_error.h",
        "vulkan_invocation.cc",
        "vulkan_invocation.h",
        "vulkan_runtime.cc",
        "vulkan_state.cc",
        "vulkan_state.h",
        "wrappers.cc",
    ],
    local_defines = ["VULKAN_RT_BUILD"],
    deps = [
        "//pmlc/compiler",
        "//pmlc/rt",
        "@half",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVLowering",
        "@llvm-project//mlir:mlir_runner_utils",
        "@volk",
    ],
    alwayslink = 1,
    linkstatic = 1,
)

plaidml_cc_binary(
    name = "vkprobe",
    srcs = ["vkprobe.cc"],
    deps = ["@volk"],
)
