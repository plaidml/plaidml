# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:tblgen.bzl", "gentbl")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "td_files",
    srcs = [
        "base.td",
        "eltwise.td",
        "ops.td",
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
        ("-gen-op-decls", "ops.h.inc"),
        ("-gen-op-defs", "ops.cc.inc"),
        ("-gen-dialect-decls -dialect=tile", "dialect.h.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    td_srcs = [":td_files"],
)

gentbl(
    name = "interfaces-gen",
    tbl_outs = [
        ("-gen-op-interface-decls", "interfaces.h.inc"),
        ("-gen-op-interface-defs", "interfaces.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "interfaces.td",
    td_srcs = ["@llvm-project//mlir:OpBaseTdFiles"],
)

plaidml_cc_library(
    name = "ir",
    srcs = [
        "dialect.cc",
        "eltwise.cc",
        "ops.cc",
        "util.cc",
    ],
    hdrs = [
        "ops.h",
        "types.h",
        "util.h",
    ],
    deps = [
        ":gen-base",
        ":gen-ops",
        ":interfaces-gen",
        "//pmlc/util",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
