# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_binary")

plaidml_cc_binary(
    name = "pmlc-opt",
    srcs = ["pmlc-opt.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/dialect/tile",
        "//pmlc/dialect/stripe",
        "//pmlc/dialect/scalar",
        "@mlir//:EDSC",
        "@mlir//:OptMain",
        "@mlir//:Parser",
        "@mlir//:TestTransforms",
        "@mlir//:Transforms",
    ],
)
