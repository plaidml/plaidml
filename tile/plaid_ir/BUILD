# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")
load("//bzl:mlir.bzl", "mlir_tblgen")

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
    name = "plaid_ir",
    hdrs = [
        "analysis.h",
        "dialect.h",
        "mlir.h",
        "ops.h",
	"padding_pass.h",
        "types.h",
        ":gen-op-defs",
    ],
    srcs = [
        "analysis.cc",
        "dialect.cc",
        "ops.cc",
	"padding_pass.cc",
        ":gen-op-decls",
    ],
    tags = ["llvm"],
    deps = [
        "@mlir//:ir",
        "@boost",
    ],
)

plaidml_cc_library(
    name = "transcode",
    hdrs = [
        "transcode.h",
    ],
    srcs = [
        "transcode.cc",
    ],
    tags = ["llvm"],
    deps = [
	":plaid_ir",
        "//tile/stripe",
    ],
)

plaidml_cc_test(
    name = "plaid_ir_test",
    srcs = ["plaid_ir_test.cc"],
    tags = ["llvm"],
    deps = [
        ":transcode",
	"//tile/lang",
	"//tile/lib",
	"//tile/codegen",
    ],
)
