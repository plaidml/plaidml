# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "tile",
    srcs = [
        "builder.cc",
        "gradient.cc",
        "gradient.h",
    ],
    hdrs = [
        "builder.h",
        "program.h",
    ],
    deps = [
        "//base/util",
        "//pmlc/dialect/eltwise",
        "//pmlc/dialect/stripe",  # TODO: remove this dep
        "//pmlc/dialect/tile/ir",
        "//pmlc/dialect/tile/transforms",
        "//pmlc/util",
        "//tile/base",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)
