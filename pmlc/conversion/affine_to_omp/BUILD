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
    name = "affine_to_omp",
    srcs = [
        "pass_detail.h",
        "affine_to_omp.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":gen-pass",
        "//pmlc/util",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:LLVMTransforms",
    ],
)
