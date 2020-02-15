# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

plaidml_cc_library(
    name = "lib",
    srcs = ["pmlc-translate.cc"],
    deps = [
        "//pmlc/dialect/tile",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateClParser",
        "@llvm-project//mlir:Translation",
    ],
)

plaidml_cc_binary(
    name = "pmlc-translate",
    deps = [":lib"],
)
