# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")
load("//vendor/mlir:mlir.bzl", "mlir_tblgen")

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
    name = "mir",
    srcs = [
        "analysis.cc",
        "dialect.cc",
        "ops.cc",
        ":gen-op-defs",
    ],
    hdrs = [
        "analysis.h",
        "attrs.h",
        "mlir.h",
        "ops.h",
        "types.h",
        ":gen-op-decls",
    ],
    tags = ["llvm"],
    deps = [
        "//pmlc/dialect/scalar",
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
        "to_mir.cc",
        "to_stripe.cc",
    ],
    hdrs = [
        "transcode.h",
    ],
    tags = ["llvm"],
    deps = [
        ":mir",
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
    tags = ["llvm"],
    deps = [
        "//tile/codegen:proto_cc",
	":mir",
    ],
)

plaidml_cc_test(
    name = "mir_test",
    srcs = ["mir_test.cc"],
    tags = ["llvm", "mir_test"],
    deps = [
        ":transcode",
        ":passes",
        "//tile/codegen",
        "//tile/lang",
        "//tile/lib",
    ],
)
