# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "x86",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//pmlc/compiler",
        "//pmlc/dialect/pxa",
        "@mlir//:AffineToStandardTransforms",
        "@mlir//:LLVMTransforms",
    ],
    alwayslink = 1,
)
