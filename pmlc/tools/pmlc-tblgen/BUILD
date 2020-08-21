# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

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

plaidml_cc_library(
    name = "lib",
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
    alwayslink = 1,
)

plaidml_cc_binary(
    name = "pmlc-tblgen",
    deps = [":lib"],
)
