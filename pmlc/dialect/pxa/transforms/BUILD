# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:tblgen.bzl", "gentbl")

package(default_visibility = ["//visibility:public"])

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
        "cache.cc",
        "dataflow_opt.cc",
        "fusion.cc",
        "localize.cc",
        "pass_detail.h",
        "resize_tmps.cc",
        "stencil.cc",
        "stencil_gemm.cc",
        "test_analysis.cc",
        "tile.cc",
        "vectorize.cc",
    ],
    hdrs = [
        "autotile.h",
        "cache.h",
        "passes.h",
        "stencil.h",
        "tile.h",
        "vectorize.h",

    ],
    deps = [
        ":gen-pass",
        "//pmlc/dialect/eltwise/ir",
        "//pmlc/dialect/pxa/analysis",
        "//pmlc/dialect/pxa/ir",
        "//pmlc/dialect/xsmm/ir",
        "@llvm-project//mlir:VectorOps",
    ],
)
