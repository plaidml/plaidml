# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:conda.bzl", "conda_binary")

conda_binary(
    name = "py_cffi",
    srcs = ["py_cffi.py"],
    env = select({
        "//toolchain:windows_x86_64": "@com_intel_plaidml_conda_tools_windows//:env",
        "//conditions:default": "@com_intel_plaidml_conda_tools_unix//:env",
    }),
)
