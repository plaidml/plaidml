# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "vulkan_runtime",
    srcs = [
        "vulkan_runtime.cc",
    ],
    hdrs = [
    	"vulkan_pfn.h",
    	"vulkan_runtime.h",
    	"vulkan_runtime_support.h",
    ],
    linkopts = [
    	"-ldl",
    ],
    deps = [
        "@vulkan_headers//:inc",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:SPIRVSerialization",
        "@llvm-project//mlir:SPIRVDialect",
    ],
    alwayslink=1,
)


plaidml_cc_test(
    name = "vulkan_runtime_tests",
    srcs = [
    	"vulkan_runtime_tests.cc",
    ],
    linkopts = [
    	"-ldl",
    ],
    deps = [
        ":vulkan_runtime",
        "//plaidml2:testenv_mlir",
        "//testing:matchers",
        "//tile/codegen",
        "@gmock//:gtest",
    ],
)