# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")
load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

plaidml_cc_library(
    name = "model",
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
    name = "pmlc-tblgen",
    srcs = [
        "oplib_gen.cc",
    ],
    deps = [
        ":model",
        "//pmlc/tools/pmlc-tblgen/cpp",
        "//pmlc/tools/pmlc-tblgen/python",
        "//pmlc/util",
        "@llvm-project//mlir:MlirTableGenMain",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TableGen",
    ],
)
