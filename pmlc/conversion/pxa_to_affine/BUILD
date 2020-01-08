# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "pxa_to_affine",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    tags = ["llvm"],
    deps = [
        "//pmlc/dialect/pxa/ir",
        "@llvm-project//mlir:AffineDialectRegistration",
        "@llvm-project//mlir:LoopDialectRegistration",
    ],
    alwayslink = 1,
)
