# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "vulkan_runner",
    srcs = [
        "vulkan_runner.cc",
    ],
    hdrs = [
        "vulkan_fn.h",
        "vulkan_runner.h",
    ],
    tags = [
        "skip_windows",
    ],
    deps = [
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVSerialization",
        "@llvm-project//mlir:Support",
        "@vulkan_headers//:inc",
        "@vulkan_loader",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "vulkan_runner_tests",
    srcs = [
        "vulkan_runner_tests.cc",
    ],
    tags = [
        "manual",
        "skip_windows",
    ],
    deps = [
        ":vulkan_runner",
        "//plaidml:testenv",
        "@gmock//:gtest",
    ],
)
