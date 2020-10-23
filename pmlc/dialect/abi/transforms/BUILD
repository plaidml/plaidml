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
        "denormalize_constants.cc",
        "lower_to_abi.cc",
        "pass_detail.h",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":gen-pass",
        "//pmlc/dialect/abi/ir",
        "//pmlc/util",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
    ],
)
