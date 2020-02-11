# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")
load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

gentbl(
    name = "gen",
    tbl_outs = [
        (
            "-gen-cpp-impl",
            "oplib_impl.h.inc",
        ),
        (
            "-gen-cpp-wrappers",
            "oplib_wrappers.h.inc",
        ),
        (
            "-gen-py-wrappers",
            "oplib_wrappers.py",
        ),
    ],
    tblgen = "//pmlc/tools/pmlc-tblgen",
    td_file = "ops.td",
)
