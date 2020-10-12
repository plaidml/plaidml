# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "llvm",
    srcs = [
        "builtin.cc",
        "device.cc",
        "device.h",
        "runtime.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":openmp",
        "//pmlc/compiler",
        "//pmlc/rt",
    ],
    alwayslink = 1,
)

plaidml_cc_library(
    name = "openmp",
    srcs = ["openmp.cc"],
    copts = ["-w"],
    deps = [
        "//pmlc/rt",
        "@llvm-project//openmp",
    ],
    alwayslink = 1,
)
