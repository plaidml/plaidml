# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "lib",
    srcs = ["pmlc-opt.cc"],
    deps = [
        "//pmlc:all_passes_and_dialects",
        "//pmlc/util",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Parser",
    ],
    alwayslink = 1,
)

plaidml_cc_binary(
    name = "pmlc-opt",
    deps = [":lib"],
)
