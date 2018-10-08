# Copyright 2017-2018 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "selector",
    srcs = ["selector.cc"],
    hdrs = ["selector.h"],
    deps = [
        "//tile/proto:hal_cc",
        "@boost//:regex",
    ],
)

plaidml_cc_library(
    name = "settings",
    srcs = ["settings.cc"],
    hdrs = ["settings.h"],
    deps = [
        "//tile/lang",
        "//tile/proto:hal_cc",
    ],
)
