# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")
load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")

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
    copts = COPTS,
    deps = [
        ":gen-ops",
        "//pmlc/util",
        "@llvm-project//llvm:support",
    ],
)

plaidml_cc_library(
    name = "ir",
    srcs = ["registration.cc"],
    copts = COPTS,
    deps = [":impl"],
    alwayslink = 1,
)
