# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "compiler",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//pmlc/conversion/tile_to_pxa",
        "//pmlc/dialect/tile/transforms",
        "//pmlc/util",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMTransforms",
    ],
)
