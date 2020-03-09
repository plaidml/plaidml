# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")
load("//tools/heatmap:build_defs.bzl", "heatmap")

plaidml_cc_library(
    name = "lib",
    srcs = [
        "autotile.cc",
        "stencil.cc",
        "test_analysis.cc",
        "tile.cc",
        ":heatmap"
    ],
    hdrs = [
        "autotile.h",
        "heatmap.h",
        "tile.h",
        "passes.h",
        "test_analysis.h",
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

heatmap(
    name = "heatmap",
    out = "heatmap.cc",
    csv = "heatmap_skx_xeonplat_8180_1-7GHz_mblocked.csv.gz",
    template = "heatmap.tpl.cc",
)
