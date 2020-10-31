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
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    td_srcs = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

plaidml_cc_library(
    name = "comp_to_llvm",
    srcs = [
        "common.cc",
        "comp_to_ocl.cc",
        "comp_to_vulkan.cc",
        "pass_detail.h",
        "utils.h",
        "utils.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":gen-pass",
        "//pmlc/dialect/comp/ir",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVSerialization",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:Support",
    ],
)
