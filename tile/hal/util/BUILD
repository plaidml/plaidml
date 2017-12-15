# Copyright 2017, Vertex.AI.

load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "selector",
    srcs = ["selector.cc"],
    hdrs = ["selector.h"],
    deps = [
        "@boost_archive//:regex",
        "//tile/proto:hal_cc",
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
