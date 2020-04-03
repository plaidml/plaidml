# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary")

plaidml_cc_binary(
    name = "conda_run",
    srcs = select({
        "@bazel_tools//src/conditions:windows": ["conda_windows.cc"],
        "//conditions:default": ["conda_unix.cc"],
    }),
    data = select({
        "@bazel_tools//src/conditions:windows": [
            "@com_intel_plaidml_conda//:conda",
            "@com_intel_plaidml_conda//:python",
        ],
        "//conditions:default": [
            "@com_intel_plaidml_conda//:python",
        ],
    }),
    deps = [
        "//pmlc/util:env",
        "@bazel_tools//tools/cpp/runfiles",
        "@boost//:filesystem",
    ] + select({
        "@bazel_tools//src/conditions:windows": ["@boost//:process"],
        "//conditions:default": [],
    }),
)
