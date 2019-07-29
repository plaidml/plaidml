# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:private"])

plaidml_cc_library(
    name = "opencl",
    srcs = ["opencl.cc"],
    hdrs = ["opencl.h"],
    copts = [
        "-DCL_USE_DEPRECATED_OPENCL_1_2_APIS",
        "-DCL_SILENCE_DEPRECATION",
    ],
    linkopts = select({
        "//toolchain:macos_x86_64": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "@opencl_headers//:inc",
    ],
)
