# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:tblgen.bzl", "gentbl")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "td_files",
    srcs = [
        "base.td",
        "//pmlc/util:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "gen-ops",
    tbl_outs = [
        ("-gen-op-decls", "ops.h.inc"),
        ("-gen-op-defs", "ops.cc.inc"),
        ("-gen-dialect-decls", "dialect.h.inc"),
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
        "ops.cc.inc",
        "util.cc",
    ],
    hdrs = [
        "ops.h",
        "types.h",
        "util.h",
    ],
    deps = [
        ":gen-ops",
        "//pmlc/util",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
)
