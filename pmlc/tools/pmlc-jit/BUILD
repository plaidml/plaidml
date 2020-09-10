# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "lib",
    srcs = ["pmlc-jit.cc"],
    deps = [
        "//pmlc:all_passes_and_dialects",
        "//pmlc/compiler",
        "//pmlc/rt",
        "//pmlc/rt/llvm",
        "//pmlc/rt/vulkan",
        "//pmlc/util",
        "@llvm-project//mlir:ExecutionEngineUtils",
    ],
)

plaidml_cc_binary(
    name = "pmlc-jit",
    deps = [":lib"],
)
