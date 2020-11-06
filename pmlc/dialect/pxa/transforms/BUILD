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
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    td_srcs = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

plaidml_cc_library(
    name = "transforms",
    srcs = [
        "autotile.cc",
        "buffer_placement.cc",
        "cache.cc",
        "cpu_thread.cc",
        "dataflow_opt.cc",
        "fusion.cc",
        "gpu_thread.cc",
        "layout_utils.cc",
        "localize.cc",
        "nest_loops.cc",
        "normalize.cc",
        "pass_detail.h",
        "reorder_layouts.cc",
        "resize_tmps.cc",
        "simplify_with_constraints.cc",
        "stencil.cc",
        "stencil_gemm.cc",
        "subgroups.cc",
        "test_analysis.cc",
        "tile.cc",
        "tile_accumulate.cc",
        "vectorize.cc",
        "vectorize_mem.cc",
    ],
    hdrs = [
        "autotile.h",
        "cache.h",
        "gpu_thread.h",
        "layout_utils.h",
        "normalize.h",
        "passes.h",
        "reorder_layouts.h",
        "simplify_with_constraints.h",
        "stencil.h",
        "tile.h",
        "tile_accumulate.h",
        "vectorize.h",
    ],
    deps = [
        ":gen-pass",
        "//pmlc/dialect/pxa/analysis",
        "//pmlc/dialect/pxa/ir",
        "//pmlc/dialect/xsmm/ir",
        "@llvm-project//mlir:VectorOps",
    ],
)
