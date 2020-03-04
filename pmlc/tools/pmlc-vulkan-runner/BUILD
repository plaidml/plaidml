# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_binary",
    "plaidml_cc_library",
    "plaidml_cc_shlib",
)

plaidml_cc_binary(
    name = "pmlc-vulkan-runner",
    srcs = [
        "pmlc-vulkan-runner.cc",
    ],
    data = [
        "addf.mlir",
    ],
    deps = [
        ":vulkan_shlib",
        "//pmlc/rt",
        "@llvm-project//mlir:MlirJitRunner",
        "@llvm-project//mlir/tools/mlir-vulkan-runner:VulkanRuntime",
        "@vulkan_loader",
    ],
)

plaidml_cc_shlib(
    name = "shlib",
    shlib_name = "vulkan_rt",
    deps = [
        "//pmlc/rt",
        "@llvm-project//mlir/tools/mlir-vulkan-runner:VulkanRuntime",
    ],
)

plaidml_cc_library(
    name = "vulkan_shlib",
    srcs = select({
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": [":shlib"],
    }),
    linkstatic = 1,
)
