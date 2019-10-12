# Copyright 2019 Intel Corporation.

load("//vendor/mlir:mlir.bzl", "COPTS", "mlir_tblgen")
load("//bzl:plaidml.bzl", "plaidml_cc_library")

exports_files([
    "enums.td",
    "interfaces.td",
])

mlir_tblgen(
    name = "gen-op-enum-decls",
    src = "enums.td",
    out = "enums.h.inc",
    action = "-gen-enum-decls",
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-enum-defs",
    src = "enums.td",
    out = "enums.cc.inc",
    action = "-gen-enum-defs",
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
    name = "util",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":gen-op-enum-decls",
        ":gen-op-enum-defs",
        ":gen-op-interface-decls",
        ":gen-op-interface-defs",
        "//base/util",
        "//tile/base",
        "@llvm//:support",
        "@mlir//:IR",
    ],
)
