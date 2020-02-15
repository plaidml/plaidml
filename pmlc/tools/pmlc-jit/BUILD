# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

plaidml_cc_library(
    name = "lib",
    srcs = ["pmlc-jit.cc"],
    linkopts = ["-ldl"],
    deps = [
        "//pmlc/rt:rt_shlib",
        "//pmlc/target/x86",
        "@llvm-project//mlir:MlirJitRunner",
    ],
)

plaidml_cc_binary(
    name = "pmlc-jit",
    deps = [":lib"],
)
