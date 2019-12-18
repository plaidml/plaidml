# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")
load("//bzl:plaidml.bzl", "plaidml_cc_library")

filegroup(
    name = "enums-td",
    srcs = ["enums.td"],
)

filegroup(
    name = "interfaces-td",
    srcs = ["interfaces.td"],
)

gentbl(
    name = "interfaces-gen",
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
    td_srcs = ["@mlir//:OpBaseTdFiles"],
)

gentbl(
    name = "enums-gen",
    tbl_outs = [
        (
            "-gen-enum-decls",
            "enums.h.inc",
        ),
        (
            "-gen-enum-defs",
            "enums.cc.inc",
        ),
    ],
    td_file = "enums.td",
    td_srcs = ["@mlir//:OpBaseTdFiles"],
)

plaidml_cc_library(
    name = "util",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    copts = COPTS,
    deps = [
        ":enums-gen",
        ":interfaces-gen",
        "//base/util",
        "//tile/base",
        "@llvm//:support",
        "@mlir//:IR",
    ],
)
