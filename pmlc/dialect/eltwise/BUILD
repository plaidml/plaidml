# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:mlir.bzl", "COPTS", "mlir_tblgen")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
)

MLIR_TBLGEN_ALSO = [
    "enums.td",
    "predicates.td",
]

mlir_tblgen(
    name = "gen-op-decls",
    src = "ops.td",
    out = "ops.h.inc",
    action = "-gen-op-decls",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-defs",
    src = "ops.td",
    out = "ops.cpp.inc",
    action = "-gen-op-defs",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-enums-decls",
    src = "ops.td",
    out = "ops_enums.h.inc",
    action = "-gen-enum-decls",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-enums-defs",
    src = "ops.td",
    out = "ops_enums.cpp.inc",
    action = "-gen-enum-defs",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-interface-decls",
    src = "ops.td",
    out = "ops_interfaces.h.inc",
    action = "-gen-op-interface-decls",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-interface-defs",
    src = "ops.td",
    out = "ops_interfaces.cpp.inc",
    action = "-gen-op-interface-defs",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

plaidml_cc_library(
    name = "eltwise",
    srcs = [
        "dialect.cc",
        "ops.cc",
        "ops.cpp.inc",
        "ops_enums.cpp.inc",
        "ops_interfaces.cpp.inc",
        "util.cc",
    ],
    hdrs = [
        "dialect.h",
        "ops.h",
        "ops.h.inc",
        "ops_enums.h.inc",
        "ops_interfaces.h.inc",
        "types.h",
        "util.h",
    ],
    copts = COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":gen-op-decls",
        ":gen-op-defs",
        "//tile/base",
        "@llvm//:support",
        "@mlir//:IR",
        "@mlir//:StandardOps",
    ],
    alwayslink = 1,
)
