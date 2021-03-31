# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "lib",
    srcs = ["pmlc-vulkan-runner.cc"],
    deps = [
        "//pmlc:all_passes_and_dialects",
        "//pmlc/rt",
        "//pmlc/rt/llvm",
        "//pmlc/rt/vulkan",
        "@llvm-project//mlir:MlirJitRunner",
    ],
)

plaidml_cc_binary(
    name = "pmlc-vulkan-runner",
    deps = [":lib"],
)
