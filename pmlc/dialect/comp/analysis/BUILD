# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "analysis",
    srcs = [
        "mem_sync_tracker.cc",
    ],
    hdrs = [
        "mem_sync_tracker.h",
    ],
    deps = [
        "//pmlc/dialect/comp/ir",
    ],
)
