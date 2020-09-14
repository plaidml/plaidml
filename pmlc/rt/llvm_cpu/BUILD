# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "llvm_cpu",
    srcs = [
        "builtin.cc",
        "runtime.cc",
    ],
    deps = [
        "//pmlc/compiler",
        "//pmlc/rt",
    ],
    alwayslink = 1,
    visibility = ["//visibility:public"],
)
