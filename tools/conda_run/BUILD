# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

plaidml_cc_binary(
    name = "conda_run",
    srcs = ["conda.cc"],
    linkopts = select({
        "@toolchain//:linux_x86_64": ["-pthread"],
        "//conditions:default": [],
    }),
    deps = [
        "//base/util",
        "@boost//:filesystem",
    ],
)
