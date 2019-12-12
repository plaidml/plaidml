# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")
load("//vendor/mlir:mlir.bzl", "COPTS", "mlir_tblgen")

MLIR_TBLGEN_ALSO = [
    "//pmlc/util:enums.td",
    "//pmlc/dialect/eltwise:predicates.td",
]

mlir_tblgen(
    name = "gen-op-decls",
    src = "ops.td",
    out = "ops.h.inc",
    action = "-gen-op-decls",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-defs",
    src = "ops.td",
    out = "ops.cc.inc",
    action = "-gen-op-defs",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

plaidml_cc_library(
    name = "pxa",
    srcs = [
        "from_tile.cc",
        "ops.cc",
        "ops.cc.inc",
    ],
    hdrs = [
        "dialect.h",
        "from_tile.h",
        "ops.h",
        "ops.h.inc",
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        ":gen-op-decls",
        ":gen-op-defs",
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

