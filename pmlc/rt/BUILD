# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_shlib")

plaidml_cc_library(
    name = "rt",
    srcs = [
        "runtime.cc",
        "memref.h",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "xsmm.cc",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
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

plaidml_cc_shlib(
    name = "shlib",
    shlib_name = "plaidml_rt",
    visibility = ["//visibility:public"],
    deps = [":rt"],
)

plaidml_cc_library(
    name = "link",
    srcs = select({
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": [":shlib"],
    }),
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
