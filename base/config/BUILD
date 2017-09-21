# Copyright 2017, Vertex.AI.
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "config",
    srcs = [
        "config.cc",
    ],
    hdrs = [
        "config.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util:util_cc",
        "//external:protobuf_clib",
    ],
)
