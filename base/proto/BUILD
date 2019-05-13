# Copyright 2019 Intel Corporation.
package(default_visibility = ["//visibility:private"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "proto",
    srcs = ["proto.cc"],
    hdrs = ["proto.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "@com_google_protobuf//:protobuf",
    ],
)
