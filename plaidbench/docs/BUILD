# Copyright 2018 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_binary")
load("//bzl:docs.bzl", "sphinx")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "sphinx",
    srcs = ["//bzl:sphinx.py"],
    deps = ["@com_intel_plaidml//plaidbench:py"],
)

sphinx(
    name = "docs",
    srcs = glob(["**/*"]),
    out = "html",
    conf = "conf.py",
    sphinx = ":sphinx",
    tags = [
        "docs",
        "manual",
    ],
)
