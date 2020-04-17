# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:tblgen.bzl", "gentbl")

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
        "pass_detail.h",
        "gpu_to_vulkan.cc",
        "kernel_outlining.cc",
        "llvm_lowering.cc",
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
