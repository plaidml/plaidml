# Copyright 2017-2018 Intel Corporation.
load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "config",
    srcs = ["config.cc"],
    hdrs = ["config.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "@com_google_protobuf//:protobuf",
    ],
)
