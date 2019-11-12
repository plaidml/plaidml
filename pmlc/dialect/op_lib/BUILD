# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:mlir.bzl", "COPTS", "mlir_tblgen")
load("//bzl:plaidml.bzl", "plaidml_cc_library")

MLIR_TBLGEN_ALSO = [
    "//pmlc/util:enums.td",
    "//pmlc/dialect/eltwise:predicates.td",
]

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

mlir_tblgen(
    name = "pml_gen_op_lib_wrappers",
    src = "ops.td",
    out = "op_lib_wrappers.h.inc",
    action = "-gen-op-lib-wrappers",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"]
)

plaidml_cc_library(
    name = "gen_op_lib_wrappers",
    srcs = [
        "OpLibWrapperGen.cpp",
    ],
    deps = [
        "@mlir//:TableGen",
    ],
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "plaidml_op_lib_wrappers",
    srcs = [
        "op_lib_wrappers.cc"
    ],
    hdrs = [
        "op_lib_wrappers.h.inc",
    ],
    copts = COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml2/core:core_mlir",
        "//plaidml2/edsl:edsl_mlir",
        "//plaidml2/op:op_mlir",
        "//pmlc/dialect/op_lib:gen_op_lib_wrappers",
        ":pml_gen_op_lib_wrappers",
    ],
    alwayslink = 1,
)
