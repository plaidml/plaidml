# Copyright 2019 Intel Corporation
#
# For build instructions, see <docs/building.md>.

package(default_visibility = ["//visibility:public"])

load("@rules_pkg//:pkg.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_runtime_pair")

pkg_tar(
    name = "pkg",
    srcs = [
        "//plaidbench:wheel",
        "//plaidml:wheel",
        "//plaidml/keras:wheel",
        "//plaidml2:wheel",
        "//plaidml2/bridge/keras:wheel",
    ],
    extension = "tar.gz",
)

py_runtime(
    name = "py3_runtime",
    files = select({
        "@com_intel_plaidml//toolchain:windows_x86_64": [
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
