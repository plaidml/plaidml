# Copyright 2020 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "gtest_main",
    srcs = ["gtest_main.cc"],
    deps = [
        "//pmlc/util",
        "@com_google_googletest//:gtest",
        "@easylogging",
        "@gflags",
    ],
)
