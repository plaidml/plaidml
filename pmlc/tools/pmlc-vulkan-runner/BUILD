# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_binary", "plaidml_cc_test")

plaidml_cc_library(
    name = "MLIRVulkanRuntime",
    srcs = [
        "VulkanRuntime.cpp",
    ],
    hdrs = [
    	"VulkanRuntime.h",
    	"VulkanRuntimeSupport.h",
    ],
    deps = [
    	"@xcd//:lib",
    	"@vulkan_sdk_lib//:lib",
        "@vulkan_headers//:inc",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:SPIRVSerialization",
        "@llvm-project//mlir:SPIRVDialect",
    ],
    alwayslink=1,
)


plaidml_cc_test(
    name = "runtime_tests",
    srcs = [
    	"VulkanRuntimeTests.cpp",
    ],
    tags = [
        "llvm",
        "windows_fail",
    ],
    deps = [
        ":MLIRVulkanRuntime",
        "@gmock//:gtest",
    ],
)


plaidml_cc_binary(
    name = "vulkan_test",
    srcs = [
    	"VulkanTest.cpp",
    ],
    deps = [
    	"@xcd//:lib",
    	"@vulkan_sdk_lib//:lib",
    	"@vulkan_headers//:inc",
    ],
)