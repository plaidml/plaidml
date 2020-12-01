# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:tblgen.bzl", "gentbl")

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

gentbl(
    name = "enums-gen",
    tbl_outs = [
        (
            "-gen-enum-decls",
            "enums.h.inc",
        ),
        (
            "-gen-enum-defs",
            "enums.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "enums.td",
    td_srcs = ["@llvm-project//mlir:OpBaseTdFiles"],
)

plaidml_cc_library(
    name = "gpu_to_comp",
    srcs = [
        "gpu_to_comp.cc",
        "pass_detail.h",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":gen-pass",
        ":enums-gen",
        "//pmlc/dialect/comp/ir",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:Support",
    ],
)
