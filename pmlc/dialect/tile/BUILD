# Copyright 2019 Intel Corporation.

load("//vendor/mlir:mlir.bzl", "COPTS", "mlir_tblgen")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
)

MLIR_TBLGEN_ALSO = [
    "interfaces.td",
    "//pmlc/util:enums.td",
    "//pmlc/util:interfaces.td",
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
    name = "tile",
    srcs = [
        "builder.cc",
        "contraction.cc",
        "dialect.cc",
        "interfaces.cc.inc",
        "lowering.cc",
        "ops.cc",
        "ops.cc.inc",
        "types.cc",
    ],
    hdrs = [
        "builder.h",
        "contraction.h",
        "dialect.h",
        "interfaces.h.inc",
        "lowering.h",
        "ops.h",
        "ops.h.inc",
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
        "//pmlc/util",
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
