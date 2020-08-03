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
    name = "comp",
    srcs = [
        "pass_detail.h",
        "gpu_to_comp.cc",
        "comp_to_opencl.cc",
    ],
    hdrs = [
        "passes.h"
    ],
    deps = [
        ":gen-pass",
        "//pmlc/dialect/comp/ir",
        "//pmlc/util",
        "@llvm-project//mlir:GPUToSPIRVTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:LLVMTransforms",
    ],
)
