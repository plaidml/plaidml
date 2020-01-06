# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")
load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "tile_to_pxa",
    srcs = [
        "pass.cc",
    ],
    hdrs = [
        "pass.h",
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        "//pmlc/dialect/pxa",
        "//pmlc/dialect/tile",
    ],
    alwayslink = 1,
)
