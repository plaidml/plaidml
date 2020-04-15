# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")
load("//vendor/mlir:tblgen.bzl", "gentbl")

filegroup(
    name = "td_files",
    srcs = [
        "interfaces.td",
        "predicates.td",
        "//pmlc/util:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "gen-interfaces",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "interfaces.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "interfaces.cc.inc",
        ),
    ],
    td_file = "interfaces.td",
    td_srcs = [":td_files"],
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
        "dialect.cc",
        "ops.cc",
        "ops.cc.inc",
        "util.cc",
    ],
    hdrs = [
        "ops.h",
        "util.h",
    ],
    deps = [
        ":gen-interfaces",
        ":gen-ops",
        "//pmlc/util",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
)
