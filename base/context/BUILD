# Copyright 2017, Vertex.AI.
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
        "//base/util:util_cc",
        "//external:boost",
        "//external:boost_thread",
        "//external:protobuf_clib",
    ],
)

plaidml_cc_test(
    name = "context_test",
    srcs = ["context_test.cc"],
    deps = [
        ":context",
        "//base/util:util_cc",
    ],
)

plaidml_cc_test(
    name = "gate_test",
    srcs = ["gate_test.cc"],
    deps = [
        ":context",
        "//base/util:util_cc",
    ],
)

plaidml_proto_library(
    name = "proto",
    srcs = ["context.proto"],
    visibility = ["//visibility:public"],
)
