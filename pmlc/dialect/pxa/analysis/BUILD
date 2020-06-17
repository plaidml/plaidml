# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "analysis",
    srcs = [
        "strides.cc",
        "uses.cc",
    ],
    hdrs = [
        "strides.h",
        "uses.h",
    ],
    deps = [
        "//pmlc/dialect/pxa/ir",
    ],
)
