# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "rt",
    srcs = [
        "boundscheck.cc",
        "builtin.cc",
        "device_id.cc",
        "executable.cc",
        "internal.h",
        "prng.cc",
        "runtime.cc",
        "runtime_registry.cc",
        "symbol_registry.cc",
        "xsmm.cc",
    ],
    hdrs = [
        "device_id.h",
        "executable.h",
        "symbol_registry.h",
        "runtime.h",
        "runtime_registry.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/compiler",
        "//pmlc/util:logging",
        "@half",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:mlir_runner_utils",
        "@xsmm",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "device_id_test",
    srcs = [
        "device_id_test.cc",
        "internal.h"
    ],
    deps = [
        ":rt",
    ],
    size = "small",
)
