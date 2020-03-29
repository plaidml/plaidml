# Copyright 2018 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("@rules_python//python:defs.bzl", "py_binary")
load("//tools/py_setup:build_defs.bzl", "py_setup")

py_binary(
    name = "bin",
    srcs = ["plaidbench.py"],
    main = "plaidbench.py",
    deps = ["//plaidbench/plaidbench"],
)

py_binary(
    name = "setup",
    srcs = ["setup.py"],
    deps = [
        "//plaidbench/plaidbench",
        "//tools/py_setup",
    ],
)

py_setup(
    name = "wheel",
    package_name = "plaidbench",
    python = select({
        "@bazel_tools//src/conditions:windows": "@com_intel_plaidml_conda_windows//:python",
        "//conditions:default": "@com_intel_plaidml_conda_unix//:python",
    }),
    tool = ":setup",
    universal = True,
    visibility = ["//visibility:public"],
)
