# Copyright 2018 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_binary")

package(default_visibility = ["//visibility:public"])

plaidml_cc_binary(
    name = "resnet50",
    srcs = ["resnet50.cc"],
    deps = [
        "//plaidml2:plaidml",
        "//pmlc/util",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)
