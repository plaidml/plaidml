# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library", "plaidml_cc_shlib")

plaidml_cc_library(
    name = "vulkan",
    srcs = [
        "vulkan_runtime.cc",
        "wrappers.cc",
    ],
    hdrs = [
        "vulkan_runtime.h",
    ],
    deps = [
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVLowering",
        "@volk",
    ],
    alwayslink = 1,
)

plaidml_cc_shlib(
    name = "shlib",
    shlib_name = "vulkan_rt",
    deps = [
        ":vulkan",
        "//pmlc/rt",
    ],
)

plaidml_cc_library(
    name = "link",
    srcs = select({
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": [":shlib"],
    }),
    linkstatic = 1,
)

plaidml_cc_binary(
    name = "vkprobe",
    srcs = ["vkprobe.cc"],
    deps = ["@volk"],
)
