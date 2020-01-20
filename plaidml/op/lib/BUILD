# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "lib",
    srcs = [
        "ops.cc",
    ],
    hdrs = [
        "ops.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml/core",
        "//plaidml/edsl",
        "//plaidml/op:api",
        "//pmlc/util",
        "@llvm-project//llvm:support",
    ],
)
