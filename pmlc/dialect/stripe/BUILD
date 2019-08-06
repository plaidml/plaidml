# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")
load("//vendor/mlir:mlir.bzl", "COPTS", "mlir_tblgen")

mlir_tblgen(
    name = "gen-op-decls",
    src = "ops.td",
    out = "ops.h.inc",
    action = "-gen-op-decls",
    also = ["//pmlc/dialect/eltwise:predicates.td"],
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-defs",
    src = "ops.td",
    out = "ops.cpp.inc",
    action = "-gen-op-defs",
    also = ["//pmlc/dialect/eltwise:predicates.td"],
    incs = ["@mlir//:include"],
)

plaidml_cc_library(
    name = "stripe",
    srcs = [
        "analysis.cc",
        "dialect.cc",
        "ops.cc",
        "ops.cpp.inc",
    ],
    hdrs = [
        "analysis.h",
        "attrs.h",
        "mlir.h",
        "ops.h",
        "ops.h.inc",
        "types.h",
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        ":gen-op-decls",
        ":gen-op-defs",
        "//pmlc/dialect/eltwise",
        "//tile/base",
        "@boost",
        "@llvm//:support",
        "@mlir//:IR",
        "@mlir//:StandardOps",
        "@mlir//:Transforms",
    ],
)

plaidml_cc_library(
    name = "transcode",
    srcs = [
        "to_stripe.cc",
        "to_stripe_mlir.cc",
    ],
    hdrs = [
        "transcode.h",
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        ":stripe",
        "//tile/stripe",
        "@mlir//:IR",
    ],
)

plaidml_cc_library(
    name = "passes",
    srcs = [
        "padding_pass.cc",
    ],
    hdrs = [
        "padding_pass.h",
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        ":stripe",
        "//tile/codegen:proto_cc",
    ],
)

plaidml_cc_test(
    name = "stripe_test",
    srcs = ["stripe_test.cc"],
    copts = COPTS,
    tags = [
        "llvm",
        "stripe_test",
    ],
    deps = [
        ":passes",
        ":transcode",
        "//tile/codegen",
        "//tile/lang",
        "//tile/lib",
    ],
)
