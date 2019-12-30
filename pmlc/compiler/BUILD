# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "compiler",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//base/util",
        "//pmlc/dialect/pxa",
        "@llvm//:support",
        "@mlir//:AffineToStandardTransforms",
        "@mlir//:ExecutionEngine",
        "@mlir//:ExecutionEngineUtils",
        "@mlir//:IR",
        "@mlir//:LLVMTransforms",
    ],
)
