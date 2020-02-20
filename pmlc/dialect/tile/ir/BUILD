# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

filegroup(
    name = "td_files",
    srcs = [
        "base.td",
        "builder.td",
        "interfaces.td",
        "ops.td",
        "//pmlc/dialect/eltwise/ir:td_files",
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
    name = "impl",
    srcs = [
        "dialect.cc",
        "ops.cc",
        "types.cc",
    ],
    hdrs = [
        "dialect.h",
        "ops.h",
        "types.h",
    ],
    copts = COPTS,
    deps = [
        ":gen-interfaces",
        ":gen-ops",
        "//pmlc/dialect/eltwise/ir",
        "//pmlc/util",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)

plaidml_cc_library(
    name = "ir",
    srcs = ["registration.cc"],
    copts = COPTS,
    deps = [":impl"],
    alwayslink = 1,
)
