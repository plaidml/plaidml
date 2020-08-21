# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "lib",
    srcs = ["pmlc-translate.cc"],
    deps = [
        "//pmlc/dialect/tile",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
)

plaidml_cc_binary(
    name = "pmlc-translate",
    deps = [":lib"],
)
