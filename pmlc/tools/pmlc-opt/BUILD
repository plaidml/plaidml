# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

plaidml_cc_library(
    name = "lib",
    srcs = ["pmlc-opt.cc"],
    deps = [
        "//pmlc:all_passes_and_dialects_no_registration",
        "//pmlc/util",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopsToGPUPass",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

plaidml_cc_binary(
    name = "pmlc-opt",
    deps = [":lib"],
)
