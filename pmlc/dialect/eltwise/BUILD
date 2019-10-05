# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:mlir.bzl", "COPTS", "mlir_tblgen")
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
    also = ["predicates.td"],
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-defs",
    src = "ops.td",
    out = "ops.cpp.inc",
    action = "-gen-op-defs",
    also = ["predicates.td"],
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-interface-decls",
    src = "ops.td",
    out = "opinterfaces.h.inc",
    action = "-gen-op-interface-decls",
    also = ["predicates.td"],
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-interface-defs",
    src = "ops.td",
    out = "opinterfaces.cpp.inc",
    action = "-gen-op-interface-defs",
    also = ["predicates.td"],
    incs = ["@mlir//:include"],
)

plaidml_cc_library(
    name = "eltwise",
    srcs = [
        "dialect.cc",
        "opinterfaces.cpp.inc",
        "ops.cc",
        "ops.cpp.inc",
        "util.cc",
    ],
    hdrs = [
        "dialect.h",
        "opinterfaces.h.inc",
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
        "//tile/base",
        "@llvm//:support",
        "@mlir//:IR",
        "@mlir//:StandardOps",
    ],
    alwayslink = 1,
)
