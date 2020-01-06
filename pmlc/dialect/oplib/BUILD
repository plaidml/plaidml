# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")
load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

plaidml_cc_library(
    name = "oplib",
    srcs = [
        "model.cc",
    ],
    hdrs = [
        "model.h",
    ],
    deps = [
        "@llvm-project//mlir:TableGen",
    ],
)

plaidml_cc_binary(
    name = "oplib-tblgen",
    srcs = [
        "oplib_gen.cc",
    ],
    deps = [
        ":oplib",
        "//base/util",
        "//pmlc/dialect/oplib/cpp",
        "//pmlc/dialect/oplib/python",
        "@llvm-project//mlir:MlirTableGenMain",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TableGen",
    ],
)

gentbl(
    name = "gen-ops",
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
    tblgen = ":oplib-tblgen",
    td_file = "ops.td",
)
