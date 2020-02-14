# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary")

plaidml_cc_binary(
    name = "pmlc-jit",
    srcs = ["pmlc-jit.cc"],
    linkopts = ["-ldl"],
    deps = [
        "//pmlc/target/x86",
        "@llvm-project//mlir:MlirJitRunner",
    ],
)
