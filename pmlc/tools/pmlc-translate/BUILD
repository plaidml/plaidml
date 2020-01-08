# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary")

plaidml_cc_binary(
    name = "pmlc-translate",
    srcs = ["pmlc-translate.cc"],
    deps = [
        "//pmlc/conversion/tile_to_stripe",
        "//pmlc/dialect/stripe:transcode",
        "//pmlc/dialect/tile",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateClParser",
        "@llvm-project//mlir:Translation",
    ],
)
