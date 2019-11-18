# Copyright 2019 Intel Corporation.

load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_binary",
    "plaidml_cc_test",
)
load("//vendor/mlir:mlir.bzl", "COPTS")

plaidml_cc_binary(
    name = "pmlc-opt",
    srcs = ["pmlc-opt.cc"],
    copts = COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/conversion/affine_to_stripe",
        "//pmlc/conversion/stripe_to_affine",
        "//pmlc/dialect/eltwise",
        "//pmlc/dialect/stripe",
        "//pmlc/dialect/stripe:passes",
        "//pmlc/dialect/tile",
        "@mlir//:EDSC",
        "@mlir//:OptMain",
        "@mlir//:Parser",
        "@mlir//:TestTransforms",
        "@mlir//:Transforms",
    ],
)

plaidml_cc_binary(
    name = "pmlc-translate",
    srcs = ["pmlc-translate.cc"],
    copts = COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/dialect/stripe:transcode",
        "//pmlc/dialect/tile",
        "@llvm//:support",
        "@mlir//:IR",
        "@mlir//:Support",
        "@mlir//:TranslateClParser",
        "@mlir//:Translation",
    ],
)

filegroup(
    name = "litfiles",
    srcs = glob(["runlit*py"]),
    visibility = ["//visibility:public"],
)

plaidml_cc_test(
    name = "test",
    srcs = ["test.cc"],
    copts = COPTS,
    deps = [
        "@mlir//:IR",
        "@mlir//:Transforms",
    ],
)
