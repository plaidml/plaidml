# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "util",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//tile/base",
        "@llvm//:support",
        "@mlir//:IR",
    ],
)
