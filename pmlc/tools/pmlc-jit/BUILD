# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

plaidml_cc_library(
    name = "lib",
    srcs = ["pmlc-jit.cc"],
    deps = [
        "//pmlc:all_passes_and_dialects_no_registration",
        "//pmlc/rt:rt_shlib",
        "//pmlc/util",
        "@llvm-project//mlir:MlirJitRunner",
    ],
)

plaidml_cc_binary(
    name = "pmlc-jit",
    deps = [":lib"],
)
