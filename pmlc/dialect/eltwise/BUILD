# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
)

filegroup(
    name = "td_files",
    srcs = [
        "interfaces.td",
        "predicates.td",
        "//pmlc/util:enums-td",
        "//pmlc/util:interfaces-td",
        "@mlir//:OpBaseTdFiles",
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
    ],
    td_file = "ops.td",
    td_srcs = [":td_files"],
)

plaidml_cc_library(
    name = "eltwise",
    srcs = [
        "dialect.cc",
        "ops.cc",
        "util.cc",
    ],
    hdrs = [
        "dialect.h",
        "ops.h",
        "types.h",
        "util.h",
    ],
    copts = COPTS,
    deps = [
        ":gen-interfaces",
        ":gen-ops",
        "//pmlc/util",
        "//tile/base",
        "@llvm//:support",
        "@mlir//:IR",
        "@mlir//:StandardOps",
    ],
    alwayslink = 1,
)
