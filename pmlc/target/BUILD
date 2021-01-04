# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "all_targets",
    defines = select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": ["PML_ENABLE_LEVEL_ZERO"],
    }),
    srcs = ["all_targets.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/compiler",
        "//pmlc/target/intel_gen",
        "//pmlc/target/intel_gen_ocl_spirv",
        "//pmlc/target/x86",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": ["//pmlc/target/intel_level_zero"],
    }),
    alwayslink = 1,
)
