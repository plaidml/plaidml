# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "lib",
    srcs = [
        "autotile.cc",
        "tile.cc",
	"test_analysis.cc"
    ],
    hdrs = [
        "autotile.h",
        "tile.h",
	"test_analysis.h"
    ],
    deps = [
        "//pmlc/dialect/pxa/ir",
        "//pmlc/dialect/pxa/analysis",
    ],
)

plaidml_cc_library(
    name = "transforms",
    srcs = ["registration.cc"],
    deps = [":lib"],
    alwayslink = 1,
)
