# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "cpp",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    deps = [
        "//pmlc/dialect/oplib",
        "@mlir//:TableGen",
    ],
)
