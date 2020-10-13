# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:tblgen.bzl", "gentbl")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "td_files",
    srcs = [
        "base.td",
        "ops.td",
        "//pmlc/dialect/eltwise/ir:td_files",
    ],
)

gentbl(
    name = "gen-base",
    tbl_outs = [
        ("-gen-enum-decls", "enums.h.inc"),
        ("-gen-enum-defs", "enums.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "base.td",
    td_srcs = [":td_files"],
)

gentbl(
    name = "gen-ops",
    tbl_outs = [
        ("-gen-dialect-decls -dialect=tile", "dialect.h.inc"),
        ("-gen-op-decls", "ops.h.inc"),
        ("-gen-op-defs", "ops.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    td_srcs = [":td_files"],
)

plaidml_cc_library(
    name = "ir",
    srcs = [
        "dialect.cc",
        "ops.cc",
    ],
    hdrs = [
        "ops.h",
    ],
    deps = [
        ":gen-base",
        ":gen-ops",
        "//pmlc/dialect/eltwise/ir",
        "//pmlc/util",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
