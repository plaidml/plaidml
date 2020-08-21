# Copyright 2020 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:tblgen.bzl", "gentbl")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "td_files",
    srcs = [
        "//pmlc/util:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

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
        (
            "-gen-dialect-decls",
            "dialect.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    td_srcs = [":td_files"],
)

plaidml_cc_library(
    name = "ir",
    srcs = ["ops.cc"],
    hdrs = ["ops.h"],
    deps = [
        ":gen-ops",
        "//pmlc/util",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
    ],
)
