# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

plaidml_cc_library(
    name = "lib",
    srcs = [
    	"pmlc-vulkan-runner.cc",
    	"multi-gpu-launch-funcs-to-vulkan-calls.cc",
    ],
    hdrs = ["multi-gpu-launch-funcs-to-vulkan-calls.h"],
    deps = [
        "//pmlc/rt/vulkan:link",
        "@llvm-project//mlir:MlirJitRunner",
    ],
)

plaidml_cc_binary(
    name = "pmlc-vulkan-runner",
    deps = [":lib"],
)
