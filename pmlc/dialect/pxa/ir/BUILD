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
    name = "interfaces-gen",
    tbl_outs = [
        ("-gen-op-interface-decls", "interfaces.h.inc"),
        ("-gen-op-interface-defs", "interfaces.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "interfaces.td",
    td_srcs = [":td_files"],
)

gentbl(
    name = "gen-ops",
    tbl_outs = [
        ("-gen-dialect-decls", "dialect.h.inc"),
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
        "interfaces.cc",
        "ops.cc",
    ],
    hdrs = [
        "interfaces.h",
        "matchers.h",
        "ops.h",
    ],
    deps = [
        ":gen-ops",
        ":interfaces-gen",
        "//pmlc/util",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
    ],
)
