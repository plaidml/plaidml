# Copyright 2018, Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_proto_library")

plaidml_proto_library(
    name = "proto",
    srcs = [
        "stripe.proto",
    ],
    visibility = ["//visibility:public"],
    deps = ["//tile/proto:shape"],
)

plaidml_cc_library(
    name = "stripe",
    srcs = [
        "stripe.cc",
    ],
    hdrs = [
        "stripe.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@boost//:filesystem",
        ":proto_cc",
        "//base/util",
        "//tile/base",
        "//tile/math",
    ],
)
