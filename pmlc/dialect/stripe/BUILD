# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

filegroup(
    name = "td_files",
    srcs = [
        "//pmlc/dialect/eltwise:td_files",
    ],
)

gentbl(
    name = "gen-ops",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "ops.cc.inc",
        ),
        (
            "-gen-op-interface-decls",
            "ops_interfaces.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "ops_interfaces.cc.inc",
        ),
    ],
    td_file = "ops.td",
    td_srcs = [":td_files"],
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
        "rewrites.h",
        "transforms.h",
        "types.h",
        "util.h",
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        ":gen-ops",
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
        "agginit_pass.cc",
        "autostencil_pass.cc",
        "padding_pass.cc",
        "populate_tensor_ref_shape_analysis.cc",
    ],
    hdrs = [
        "agginit_pass.h",
        "autostencil_pass.h",
        "nop_pass.h",
        "padding_pass.h",
        "populate_tensor_ref_shape_analysis.h",
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        ":stripe",
        ":transcode",
        "//tile/codegen:proto_cc",
        "//tile/targets/cpu",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "translate_test",
    srcs = ["translate_test.cc"],
    copts = COPTS,
    tags = [
        "llvm",
        "windows_fail",
    ],
    deps = [
        ":passes",
        ":transcode",
        "//plaidml2:testenv_mlir",
        "//testing:matchers",
        "//tile/codegen",
    ],
)
