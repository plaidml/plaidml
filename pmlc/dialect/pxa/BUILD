# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")
load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")

gentbl(
    name = "gen-ops",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "ops.cc.inc",
        ),
    ],
    td_file = "ops.td",
    td_srcs = [
        "//pmlc/dialect/eltwise:td_files",
    ],
)

plaidml_cc_library(
    name = "pxa",
    srcs = [
        "from_tile.cc",
        "ops.cc",
        "to_affine.cc",
    ],
    hdrs = [
        "dialect.h",
        "ops.h",
        "passes.h",
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        ":gen-ops",
        "//pmlc/dialect/eltwise",
        "//pmlc/dialect/tile",
        "//tile/base",
        "@boost",
        "@llvm//:support",
        "@mlir//:IR",
        "@mlir//:Parser",
        "@mlir//:StandardOps",
        "@mlir//:Transforms",
    ],
    alwayslink = 1,
)
