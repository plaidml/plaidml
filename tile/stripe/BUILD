# Copyright 2018, Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test", "plaidml_proto_library")

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
    srcs = glob(
        [
            "*.cc",
            "impl.h",
        ],
        exclude = glob(["*_test.cc"]),
    ),
    hdrs = ["stripe.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":proto_cc",
        "//base/util",
        "//tile/base",
        "//tile/math",
        "@boost",
        "@boost//:filesystem",
    ],
)

plaidml_cc_test(
    name = "stripe_test",
    srcs = ["stripe_test.cc"],
    deps = [
        ":stripe",
        "//base/util",
    ],
)
