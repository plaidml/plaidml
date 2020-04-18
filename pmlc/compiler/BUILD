# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "compiler",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//pmlc/util",
        "@llvm-project//llvm:execution_engine",
        "@llvm-project//llvm:mcjit",
        "@llvm-project//llvm:target",
        "@llvm-project//llvm:x86_code_gen",
        "@llvm-project//llvm:x86_disassembler",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:TargetLLVMIR",
    ],
)
