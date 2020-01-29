# Copyright 2020 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "transforms",
    srcs = [
	"autotile.cc",
	"tile.cc",
    ],
    hdrs = [
        "autotile.h",
        "tile.h",
        "passes.h",
    ],
    deps = [
        "//pmlc/dialect/pxa/ir",
    ],
    alwayslink=1,
)
