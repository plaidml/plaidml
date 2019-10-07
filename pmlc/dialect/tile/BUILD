# Copyright 2019 Intel Corporation.

load("//vendor/mlir:mlir.bzl", "COPTS", "mlir_tblgen")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
)

MLIR_TBLGEN_ALSO = [
    "//pmlc/dialect/eltwise:enums.td",
    "//pmlc/dialect/eltwise:predicates.td",
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
    name = "tile",
    srcs = [
        "builder.cc",
        "contraction.cc",
        "dialect.cc",
        "lowering.cc",
        "ops.cc",
        "ops.cpp.inc",
        "ops_interfaces.cpp.inc",
    ],
    hdrs = [
        "builder.h",
        "contraction.h",
        "dialect.h",
        "lowering.h",
        "ops.h",
        "ops.h.inc",
        "ops_interfaces.h.inc",
        "program.h",
        "types.h",
    ],
    copts = COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":gen-op-decls",
        ":gen-op-defs",
        ":gen-op-interface-decls",
        ":gen-op-interface-defs",
        "//base/util",
        "//pmlc/dialect/eltwise",
        "//pmlc/dialect/stripe",
        "//pmlc/dialect/stripe:transcode",
        "//tile/base",
        "//tile/lang",
        "@boost",
        "@llvm//:support",
        "@mlir//:AffineOps",
        "@mlir//:IR",
        "@mlir//:StandardOps",
    ],
    alwayslink = 1,
)
