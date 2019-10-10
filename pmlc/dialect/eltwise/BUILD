# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:mlir.bzl", "COPTS", "mlir_tblgen")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
)

MLIR_TBLGEN_ALSO = [
    "//pmlc/util:interfaces.td",
    "interfaces.td",
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
    out = "ops.cc.inc",
    action = "-gen-op-defs",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-interface-decls",
    src = "interfaces.td",
    out = "interfaces.h.inc",
    action = "-gen-op-interface-decls",
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-interface-defs",
    src = "interfaces.td",
    out = "interfaces.cc.inc",
    action = "-gen-op-interface-defs",
    incs = ["@mlir//:include"],
)

plaidml_cc_library(
    name = "eltwise",
    srcs = [
        "dialect.cc",
        "interfaces.cc.inc",
        "ops.cc",
        "ops.cc.inc",
        "util.cc",
    ],
    hdrs = [
        "dialect.h",
        "interfaces.h.inc",
        "ops.h",
        "ops.h.inc",
        "types.h",
        "util.h",
    ],
    copts = COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":gen-op-decls",
        ":gen-op-defs",
        "//pmlc/util",
        "//tile/base",
        "@llvm//:support",
        "@mlir//:IR",
        "@mlir//:StandardOps",
    ],
    alwayslink = 1,
)
