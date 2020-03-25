# Copyright 2019 Intel Corporation
#
# For build instructions, see <docs/building.md>.

package(default_visibility = ["//visibility:public"])

load("@rules_pkg//:pkg.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_runtime_pair")

exports_files([
    "LICENSE",
])

config_setting(
    name = "clang",
    values = {
        "define": "compiler=clang",
    },
)

config_setting(
    name = "gcc",
    values = {
        "define": "compiler=gcc",
    },
)

config_setting(
    name = "msvc",
    values = {
        "define": "compiler=msvc",
    },
)

pkg_tar(
    name = "pkg",
    srcs = [
        "//plaidbench:wheel",
        "//plaidml:wheel",
        "//plaidml/bridge/keras:wheel",
    ],
    extension = "tar.gz",
)

py_runtime(
    name = "py3_runtime",
    files = select({
        "@bazel_tools//src/conditions:windows": [
            "@com_intel_plaidml_conda_windows//:python",
        ],
        "//conditions:default": [
            "@com_intel_plaidml_conda_unix//:python",
        ],
    }),
    interpreter = "//tools/conda_run",
    python_version = "PY3",
)

py_runtime_pair(
    name = "py_runtime_pair",
    py3_runtime = ":py3_runtime",
)

toolchain(
    name = "py_toolchain",
    toolchain = ":py_runtime_pair",
    toolchain_type = "@rules_python//python:toolchain_type",
)
