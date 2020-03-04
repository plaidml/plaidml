# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary")

plaidml_cc_binary(
    name = "pmlc-vulkan-runner",
    srcs = [
        "pmlc-vulkan-runner.cc",
    ],
    data = [
        "addf.mlir",
    ],
    deps = [
        "//pmlc/rt/vulkan:link",
        "@llvm-project//mlir:MlirJitRunner",
    ],
)
