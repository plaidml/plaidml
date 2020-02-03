# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "stdx_to_llvm",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//pmlc/dialect/stdx/ir",
        "@llvm-project//mlir:LLVMTransforms",
    ],
    alwayslink = 1,
)
