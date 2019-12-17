# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:mlir.bzl", "COPTS", "mlir_tblgen")
load("//pmlc/dialect/op_lib:op_lib.bzl", "COPTS", "op_lib_tblgen")
load("//bzl:plaidml.bzl", "plaidml_cc_library")

MLIR_TBLGEN_ALSO = [
    "//pmlc/util:enums.td",
    "//pmlc/dialect/eltwise:predicates.td",
]

PLATFORM_COPTS = select({
    "@com_intel_plaidml//toolchain:macos_x86_64": [
        "-std=c++14",
        "-w",
    ],
    "@com_intel_plaidml//toolchain:windows_x86_64": [
        "/w",
        "/DWIN32_LEAN_AND_MEAN",
        "/std:c++17",  # This MUST match all other compilation units
    ],
    "//conditions:default": [
        "-std=c++14",
        "-w",
    ],
})

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
    name = "op-lib-tblgen",
    srcs = [
        "//pmlc/dialect/op_lib:OpLibGen.cc"
    ],
    copts = PLATFORM_COPTS,
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
    visibility = ["//visibility:public"],
    deps = [
        ":OpModel",
        ":OpCPP",
        ":OpPY",
        "//base/util",
        "@mlir//:TableGen",
        "@mlir//:TableGenTools",
        "@mlir//:Support",
    ],
)

mlir_tblgen(
    name = "pml_gen_op_interface_decls",
    src = "ops.td",
    out = "op_interfaces.h.inc",
    action = "-gen-op-interface-decls",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "pml_gen_op_interface_defs",
    src = "ops.td",
    out = "op_interfaces.cc.inc",
    action = "-gen-op-interface-defs",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "pml_gen_op_decls",
    src = "ops.td",
    out = "ops.h.inc",
    action = "-gen-op-decls",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "pml_gen_op_defs",
    src = "ops.td",
    out = "ops.cc.inc",
    action = "-gen-op-defs",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

op_lib_tblgen(
    name = "pml_gen_op_lib_cpp_impl",
    src = "ops.td",
    out = "op_lib_impl.h.inc",
    action = "-gen-op-lib-cpp-impl",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"]
)

op_lib_tblgen(
    name = "pml_gen_op_lib_cpp_wrappers",
    src = "ops.td",
    out = "op_lib_wrappers.h.inc",
    action = "-gen-op-lib-cpp-wrappers",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"]
)

op_lib_tblgen(
    name = "pml_gen_op_lib_py_wrappers",
    src = "ops.td",
    out = "op_lib_wrappers.py",
    action = "-gen-op-lib-py-wrappers",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"]
)

plaidml_cc_library(
    name = "gen_op_lib_dialect",
    srcs = [
        "OpLibGen.cc",
    ],
    deps = [
        "//base/util",
        ":OpModel",
        ":OpCPP",
        ":OpPY",
    ],
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "plaidml_op_lib_dialect",
    srcs = [
        "ops.cc"
    ],
    hdrs = [
        "op_lib_wrappers.h.inc",
        "op_lib_impl.h.inc",
    ],
    copts = COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml2/core:core_mlir",
        "//plaidml2/edsl:edsl_mlir",
        "//plaidml2/op:op_mlir",
        "//pmlc/dialect/op_lib:gen_op_lib_dialect",
        ":pml_gen_op_lib_cpp_impl",
        ":pml_gen_op_lib_cpp_wrappers",
        ":pml_gen_op_lib_py_wrappers",
    ],
    alwayslink = 1,
)
