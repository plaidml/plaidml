# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_test")

filegroup(
    name = "litfiles",
    srcs = glob(["runlit*py"]),
    visibility = ["//visibility:public"],
)

plaidml_cc_test(
    name = "test",
    srcs = ["test.cc"],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "pmlc",
    hdrs = [
        "all_dialects.h",
        "all_passes.h",
    ],
    visibility = ["//visibility:public"],
)
