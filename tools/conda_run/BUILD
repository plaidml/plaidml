# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary")

plaidml_cc_binary(
    name = "conda_run",
    srcs = select({
        "//toolchain:windows_x86_64": ["conda_windows.cc"],
        "//conditions:default": ["conda_unix.cc"],
    }),
    deps = [
        "//base/util",
        "@bazel_tools//tools/cpp/runfiles",
        "@boost//:filesystem",
    ],
)
