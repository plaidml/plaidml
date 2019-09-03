# Copyright 2018 Intel Corporation.

load("//bzl:docs.bzl", "sphinx")
load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "sphinx",
    srcs = ["//bzl:sphinx.py"],
    deps = ["//plaidml/keras"],
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
