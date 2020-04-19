# Copyright 2020 Intel Corporation

load("@io_bazel_rules_jsonnet//jsonnet:jsonnet.bzl", "jsonnet_to_json")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_shlib")

plaidml_cc_library(
    name = "rt",
    srcs = [
        "boundscheck.cc",
        "memref.h",
        "runtime.cc",
        "xsmm.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/compiler",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:mlir_runner_utils",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "@xsmm",
        ],
    }),
    alwayslink = 1,
)
