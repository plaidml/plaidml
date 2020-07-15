# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:tblgen.bzl", "gentbl")

package(default_visibility = ["//visibility:public"])

gentbl(
    name = "gen-pass",
    tbl_outs = [
        (
            "-gen-pass-decls",
            "passes.h.inc",
        ),
    ],
    td_file = "passes.td",
    td_srcs = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

plaidml_cc_library(
    name = "gpu",
    srcs = [
        "gpu_to_vulkan.cc",
        "kernel_outlining.cc",
        "pass_detail.h",
    ],
    hdrs = [
        "lowering.h",
    ],
    deps = [
        ":gen-pass",
        "//pmlc/util",
        "@llvm-project//mlir:GPUToSPIRVTransforms",
        "@llvm-project//mlir:GPUTransforms",
    ],
)
