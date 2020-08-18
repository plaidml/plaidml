# Copyright 2020 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "demo",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//pmlc/compiler",
        "//pmlc/conversion/pxa_to_affine",
        "//pmlc/conversion/tile_to_pxa",
        "//pmlc/dialect/tile/transforms",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:LLVMTransforms",
    ],
)
