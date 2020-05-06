# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")
load("//vendor/mlir:tblgen.bzl", "gentbl")

gentbl(
    name = "gen-pass",
    tbl_outs = [
        (
            "-gen-pass-decls",
            "passes.h.inc",
        ),
    ],
    td_file = "passes.td",
    td_srcs = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

plaidml_cc_library(
    name = "transforms",
    srcs = [
        "autotile.cc",
        "pass_detail.h",
        "stencil.cc",
        "test_analysis.cc",
        "tile.cc",
    ],
    hdrs = [
        "autotile.h",
        "passes.h",
        "tile.h",
        "stencil.h",
    ],
    deps = [
        ":gen-pass",
        "//pmlc/dialect/eltwise/ir",
        "//pmlc/dialect/pxa/analysis",
        "//pmlc/dialect/pxa/ir",
        "//pmlc/dialect/xsmm/ir",
    ],
)
