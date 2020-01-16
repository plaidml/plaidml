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
    ],
    copts = [
    	"-Iexternal/vulkan_sdk_include",
    ],
    deps = [
    	"@vulkan_sdk_include//:inc",
    	"@vulkan_sdk_lib//:lib",
    	"@xcd//:lib",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:SPIRVSerialization",
        "@llvm-project//mlir:SPIRVDialect",
    ],
    alwayslink=1,
)


plaidml_cc_binary(
    name = "runtime_tests",
    srcs = [
    	"RuntimeTests.cpp",
    	"Runtime.h",
    ],
    copts = COPTS,
    tags = [
        "llvm",
        "windows_fail",
    ],
    deps = [
        ":MLIRVulkanRuntime",
        "@llvm-project//mlir:SPIRVDialect",
    ],
)