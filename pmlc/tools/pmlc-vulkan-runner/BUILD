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
    	"vulkan_runner_support.h",
    ],
    linkopts = [
    	"-ldl",
    ],
    deps = [
        "@vulkan_headers//:inc",
        "@vulkan_loader//:vulkan_loader",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:SPIRVSerialization",
        "@llvm-project//mlir:SPIRVDialect",
    ],
    alwayslink=1,
)


plaidml_cc_test(
    name = "vulkan_runner_tests",
    srcs = select({
        "//toolchain:linux_x86_64": ["vulkan_runner_tests.cc"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "//toolchain:linux_x86_64": ["-ldl"],
        "//conditions:default": [],
    }),
    deps = [
        ":vulkan_runner",
        "//plaidml2:testenv_mlir",
        "//testing:matchers",
        "//tile/codegen",
        "@gmock//:gtest",
    ],
)
