# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:tblgen.bzl", "gentbl")

package(default_visibility = ["//visibility:public"])

gentbl(
    name = "gen-pass",
    tbl_outs = [
        ("-gen-pass-decls", "passes.h.inc"),
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
        "dead_memref_elimination.cc",
        "loop_unroll.cc",
        "memref_dataflow_opt.cc",
        "pass_detail.h",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":gen-pass",
        "@llvm-project//mlir:VectorOps",
    ],
)
