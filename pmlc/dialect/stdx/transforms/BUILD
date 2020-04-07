# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "lib",
    srcs = [
        "boundscheck.h",
        "boundscheck.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        "//pmlc/dialect/pxa/analysis",
        "//pmlc/dialect/pxa/ir",
        "//pmlc/dialect/tile/ir",
        "//pmlc/dialect/eltwise/ir",
    ],
)

plaidml_cc_library(
    name = "transforms",
    srcs = ["registration.cc"],
    deps = [":lib"],
    alwayslink = 1,
)
