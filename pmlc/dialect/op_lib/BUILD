# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")
load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "OpModel",
    srcs = [
        "OpModel.cc",
    ],
    hdrs = [
        "OpModel.h",
    ],
    deps = [
        "@mlir//:TableGen",
    ],
)

plaidml_cc_library(
    name = "OpCPP",
    srcs = glob([
        "cpp/ImplGen.cc",
        "cpp/WrapperGen.cc",
    ]),
    hdrs = glob([
        "cpp/ImplGen.h",
        "cpp/WrapperGen.h",
        "cpp/utils.h",
    ]),
    deps = [
        ":OpModel",
        "@mlir//:TableGen",
    ],
)

plaidml_cc_library(
    name = "OpPY",
    srcs = glob([
        "python/WrapperGen.cc",
    ]),
    hdrs = glob([
        "python/WrapperGen.h",
        "python/utils.h",
    ]),
    deps = [
        ":OpModel",
        "@mlir//:TableGen",
    ],
)

cc_binary(
    name = "oplib-tblgen",
    srcs = [
        "//pmlc/dialect/op_lib:OpLibGen.cc"
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
        ":OpModel",
        ":OpCPP",
        ":OpPY",
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
            "-gen-op-lib-cpp-impl",
            "op_lib_impl.h.inc",
        ),
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
    name = "gen_ops",
    srcs = [
        "OpLibGen.cc",
    ],
    deps = [
        "//base/util",
        ":OpModel",
        ":OpCPP",
        ":OpPY",
    ],
)

plaidml_cc_library(
    name = "plaidml_ops_cc",
    srcs = [
        "ops.cc"
    ],
    hdrs = [
        "op_lib_wrappers.h.inc",
        "op_lib_impl.h.inc",
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
