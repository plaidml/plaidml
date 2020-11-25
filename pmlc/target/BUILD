# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "all_targets",
    srcs = ["all_targets.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/compiler",
        "//pmlc/target/intel_gen",
        "//pmlc/target/intel_gen_ocl_spirv",
        "//pmlc/target/x86",
    ],
    alwayslink = 1,
)
