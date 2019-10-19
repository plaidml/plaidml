# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_binary")
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
        "//pmlc/dialect/tile",
        "@mlir//:EDSC",
        "@mlir//:OptMain",
        "@mlir//:Parser",
        "@mlir//:TestTransforms",
        "@mlir//:Transforms",
    ],
)

filegroup(
    name = "litfiles",
    srcs = glob(["runlit*py"]),
    visibility = ["//visibility:public"],
)
