# Copyright 2019 Intel Corporation.

load("//vendor/mlir:mlir.bzl", "mlir_tblgen")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
)

mlir_tblgen(
    name = "gen-op-decls",
    src = "ops.td",
    out = "ops.h.inc",
    action = "-gen-op-decls",
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-defs",
    src = "ops.td",
    out = "ops.cpp.inc",
    action = "-gen-op-defs",
    incs = ["@mlir//:include"],
)

plaidml_cc_library(
    name = "tile",
    srcs = [
        "dialect.cc",
        "ops.cc",
        ":gen-op-defs",
    ],
    hdrs = [
        "ops.h",
        ":gen-op-decls",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//tile/base",
        "//tile/lang",
        "@boost",
        "@llvm//:support",
        "@mlir//:IR",
        "@mlir//:StandardOps",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "tile_test",
    srcs = ["tile_test.cc"],
    deps = [
        ":tile",
        "//tile/codegen",
        "//tile/lang",
        "//tile/lib",
        "@mlir//:EDSC",
        "@mlir//:Transforms",
    ],
)
