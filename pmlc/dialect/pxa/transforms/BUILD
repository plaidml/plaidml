# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "lib",
    srcs = [
        "autotile.cc",
        "tile.cc",
    ],
    hdrs = [
        "autotile.h",
        "tile.h",
    ],
    deps = [
        "//pmlc/dialect/pxa/ir",
    ],
)

plaidml_cc_library(
    name = "transforms",
    srcs = ["registration.cc"],
    deps = [":lib"],
    alwayslink = 1,
)
