# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:tblgen.bzl", "gentbl")
load("//bzl:plaidml.bzl", "plaidml_cc_library")

filegroup(
    name = "td_files",
    srcs = [
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
    td_file = "ops.td",
    td_srcs = [":td_files"],
)

plaidml_cc_library(
    name = "ir",
    srcs = [
        "ops.cc",
        "types.cc",
    ],
    hdrs = [
        "ops.h",
        "types.h",
    ],
    deps = [
        ":gen-ops",
        "//pmlc/util",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
)
