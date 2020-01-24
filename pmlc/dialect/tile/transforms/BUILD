# Copyright 2020 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "lib",
    srcs = [
        "contraction.cc",
        "constant_types.cc"
    ],
    hdrs = [
        "contraction.h",
        "constant_types.h",
        "passes.h",
    ],
    deps = [
        "//pmlc/dialect/tile/ir",
        "//pmlc/util",
        "//pmlc/util/bilp",
        "//pmlc/util/math",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)

plaidml_cc_library(
    name = "transforms",
    srcs = ["registration.cc"],
    deps = [":lib"],
    alwayslink = 1,
)
