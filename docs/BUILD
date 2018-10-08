# Copyright 2018 Intel Corporation.

load("//bzl:conda.bzl", "conda_binary")
load("//bzl:docs.bzl", "sphinx")

package(default_visibility = ["//visibility:public"])

conda_binary(
    name = "sphinx",
    srcs = ["//bzl:sphinx.py"],
    env = "@com_intel_plaidml_conda_sphinx//:env",
    deps = ["//plaidml/keras"],
)

sphinx(
    name = "docs",
    srcs = glob(["**/*"]),
    out = "html",
    conf = "conf.py",
    sphinx = ":sphinx",
    tags = ["docs"],
)
