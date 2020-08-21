# Copyright 2020 Intel Corporation.

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
        "constant_types.cc",
        "contraction.cc",
        "make_program.cc",
        "padding.cc",
        "pass_detail.h",
    ],
    hdrs = [
        "contraction.h",
        "padding.h",
        "passes.h",
    ],
    deps = [
        ":gen-pass",
        "//pmlc/dialect/tile/ir",
        "//pmlc/util",
        "//pmlc/util/bilp",
        "//pmlc/util/math",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
