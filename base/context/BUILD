# Copyright 2017-2018 Intel Corporation.
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test", "plaidml_proto_library")

plaidml_cc_library(
    name = "context",
    srcs = [
        "context.cc",
        "eventlog.cc",
        "gate.cc",
    ],
    hdrs = [
        "context.h",
        "eventlog.h",
        "gate.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":proto_cc",
        "//base/util",
        "@boost",
        "@boost//:thread",
    ],
)

plaidml_cc_test(
    name = "context_test",
    srcs = ["context_test.cc"],
    deps = [
        ":context",
        "//base/util",
    ],
)

plaidml_cc_test(
    name = "gate_test",
    srcs = ["gate_test.cc"],
    deps = [
        ":context",
        "//base/util",
    ],
)

plaidml_proto_library(
    name = "proto",
    srcs = ["context.proto"],
    visibility = ["//visibility:public"],
)
