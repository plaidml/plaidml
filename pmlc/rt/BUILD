# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "rt",
    srcs = [
        "boundscheck.cc",
        "builtin.cc",
        "device.h",
        "device_id.cc",
        "executable.cc",
        "internal.h",
        "jit_executable.cc",
        "jit_executable.h",
        "prng.cc",
        "runtime_registry.cc",
        "symbol_registry.cc",
        "xsmm.cc",
    ],
    hdrs = [
        "device_id.h",
        "executable.h",
        "runtime.h",
        "runtime_registry.h",
        "symbol_registry.h",
    ],
    copts = ["-w"],
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/compiler",
        "//pmlc/util:logging",
        "@half",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:MCJIT",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:X86AsmParser",
        "@llvm-project//llvm:X86CodeGen",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:TargetLLVMIR",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:mlir_runner_utils",
        "@xsmm",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "device_id_test",
    size = "small",
    srcs = [
        "device_id_test.cc",
        "internal.h",
    ],
    deps = [
        ":rt",
    ],
)
