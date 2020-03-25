# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//tools/heatmap:build_defs.bzl", "heatmap")

heatmap(
    name = "heatmap",
    out = "heatmap.cc",
    csv = "heatmap_skx_xeonplat_8180_1-7GHz_mblocked.csv.gz",
    template = "heatmap.tpl.cc",
)

plaidml_cc_library(
    name = "x86",
    srcs = [
        "heatmap_impl.cc",
        "pipeline.cc",
        "trace_linking.cc",
        "xsmm_lowering.cc",
        ":heatmap",
    ],
    hdrs = glob(["*.h"]),
    deps = [
        "//pmlc/compiler",
        "//pmlc/conversion/pxa_to_affine",
        "//pmlc/conversion/stdx_to_llvm",
        "//pmlc/conversion/tile_to_pxa",
        "//pmlc/dialect/pxa/analysis",
        "//pmlc/dialect/pxa/transforms",
        "//pmlc/dialect/tile/transforms",
        "//pmlc/dialect/xsmm",
        "//pmlc/rt",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:LLVMTransforms",
    ],
    alwayslink = 1,
)
