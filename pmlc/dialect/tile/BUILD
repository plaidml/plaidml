# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

filegroup(
    name = "td_files",
    srcs = [
        "interfaces.td",
        "//pmlc/dialect/eltwise:td_files",
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
    name = "tile",
    srcs = [
        "builder.cc",
        "contraction.cc",
        "dialect.cc",
        "gradient.cc",
        "lowering.cc",
        "ops.cc",
        "types.cc",
    ],
    hdrs = [
        "builder.h",
        "contraction.h",
        "dialect.h",
        "gradient.h",
        "lowering.h",
        "ops.h",
        "program.h",
        "types.h",
    ],
    copts = COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":gen-interfaces",
        ":gen-ops",
        "//base/util",
        "//pmlc/dialect/eltwise",
        "//pmlc/dialect/stripe",
        "//pmlc/dialect/stripe:transcode",
        "//pmlc/util",
        "//tile/base",
        "//tile/lang",
        "@boost",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AffineOps",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
    alwayslink = 1,
)
