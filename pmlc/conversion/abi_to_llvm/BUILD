# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:tblgen.bzl", "gentbl")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "abi_to_llvm",
    srcs = [
        "abi_to_llvm.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        "//pmlc/dialect/abi/ir",
        "@llvm-project//mlir:LLVMTransforms",
    ],
)
