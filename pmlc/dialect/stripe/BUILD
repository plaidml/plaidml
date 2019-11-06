# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")
load("//vendor/mlir:mlir.bzl", "COPTS", "mlir_tblgen")

MLIR_TBLGEN_ALSO = [
    "//pmlc/util:enums.td",
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
    src = "ops.td",
    out = "ops_interfaces.h.inc",
    action = "-gen-op-interface-decls",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

mlir_tblgen(
    name = "gen-op-interface-defs",
    src = "ops.td",
    out = "ops_interfaces.cc.inc",
    action = "-gen-op-interface-defs",
    also = MLIR_TBLGEN_ALSO,
    incs = ["@mlir//:include"],
)

plaidml_cc_library(
    name = "stripe",
    srcs = [
        "affine_poly.cc",
        "analysis.cc",
        "demo_vectorize.cc",
        "dialect.cc",
        "jigsaw_pass.cc",
        "ops.cc",
        "ops.cc.inc",
        "ops_interfaces.cc.inc",
        "rewrites.cc",
        "transforms.cc",
        "types.cc",
        "util.cc",
    ],
    hdrs = [
        "affine_poly.h",
        "analysis.h",
        "dialect.h",
        "mlir.h",
        "ops.h",
        "ops.h.inc",
        "ops_interfaces.h.inc",
	"rewrites.h",
        "transforms.h",
        "types.h",
        "util.h",
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
        "@mlir//:Parser",
        "@mlir//:StandardOps",
        "@mlir//:Transforms",
    ],
    alwayslink = 1,
)

plaidml_cc_library(
    name = "transcode",
    srcs = [
        "from_mlir.cc",
        "into_mlir.cc",
    ],
    hdrs = [
        "transcode.h",
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        ":stripe",
        "//pmlc/util",
        "//tile/stripe",
        "@mlir//:IR",
        "@mlir//:Translation",
    ],
    alwayslink = 1,
)

plaidml_cc_library(
    name = "passes",
    srcs = [
        "padding_pass.cc",
        "populate_tensor_ref_shape_analysis.cc",
    ],
    hdrs = [
        "nop_pass.h",
        "padding_pass.h",
        "populate_tensor_ref_shape_analysis.h",
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        ":stripe",
        "//tile/codegen:proto_cc",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "stripe_test",
    srcs = ["stripe_test.cc"],
    copts = COPTS,
    tags = [
        "llvm",
        "windows_fail",
    ],
    deps = [
        ":passes",
        ":transcode",
        "//testing:matchers",
        "//tile/codegen",
        "//tile/lang",
        "//tile/lib",
    ],
)
