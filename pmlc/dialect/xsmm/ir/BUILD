# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")
load("//vendor/mlir:tblgen.bzl", "gentbl")

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
    ],
    td_file = "ops.td",
    td_srcs = [":td_files"],
)

plaidml_cc_library(
    name = "impl",
    srcs = [
        "dialect.cc",
        "ops.cc",
    ],
    hdrs = [
        "dialect.h",
        "ops.h",
    ],
    deps = [
        ":gen-ops",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)

plaidml_cc_library(
    name = "ir",
    srcs = ["registration.cc"],
    deps = [":impl"],
    alwayslink = 1,
)
