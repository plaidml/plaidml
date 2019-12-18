# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")
load("//bzl:plaidml.bzl", "plaidml_cc_library")

cc_binary(
    name = "oplib-tblgen",
    srcs = [
        "OpLibWrapperGen.cpp",
    ],
    includes = ["@mlir//:include"],
    linkopts = select({
        "@com_intel_plaidml//toolchain:windows_x86_64": [],
        "@com_intel_plaidml//toolchain:macos_x86_64": [],
        "//conditions:default": [
            "-pthread",
            "-ldl",
            "-lm",
        ],
    }),
    deps = [
        "//base/util",
        "@mlir//:MlirTableGenMain",
        "@mlir//:Support",
        "@mlir//:TableGen",
    ],
)

gentbl(
    name = "gen-ops",
    tbl_outs = [
        (
            "-gen-op-lib-cpp-wrappers",
            "op_lib_wrappers.h.inc",
        ),
        (
            "-gen-op-lib-py-wrappers",
            "op_lib_wrappers.py",
        ),
    ],
    tblgen = ":oplib-tblgen",
    td_file = "ops.td",
)

plaidml_cc_library(
    name = "gen_op_lib_wrappers",
    srcs = [
        "OpLibWrapperGen.cpp",
    ],
    deps = [
        "//base/util",
        "@mlir//:TableGen",
    ],
)

plaidml_cc_library(
    name = "wrappers_cc",
    srcs = [
        "op_lib_wrappers.cc",
    ],
    copts = COPTS,
    deps = [
        ":gen-ops",
        "//plaidml2/core:core_mlir",
        "//plaidml2/edsl:edsl_mlir",
        "//plaidml2/op:op_mlir",
    ],
    alwayslink = 1,
)
