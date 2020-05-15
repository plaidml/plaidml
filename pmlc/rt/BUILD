# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "rt",
    srcs = [
        "boundscheck.cc",
        "runtime.cc",
        "xsmm.cc",
        "prng.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/compiler",
        "@half",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:mlir_runner_utils",
        "@xsmm",
    ],
    alwayslink = 1,
)
