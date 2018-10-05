# Copyright 2017-2018 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_proto_library")

plaidml_proto_library(
    name = "proto",
    srcs = ["tile.proto"],
    visibility = ["//visibility:public"],
    deps = [":shape"],
)

plaidml_proto_library(
    name = "shape",
    srcs = ["shape.proto"],
    visibility = ["//visibility:public"],
)

plaidml_proto_library(
    name = "metadata",
    srcs = ["metadata.proto"],
    visibility = ["//visibility:public"],
    deps = [":shape"],
)

plaidml_proto_library(
    name = "hal",
    srcs = ["hal.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":proto",
        "//tile/lang:proto",
    ],
)

plaidml_proto_library(
    name = "schedule",
    srcs = ["schedule.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":proto",
        "//tile/lang:proto",
    ],
)

plaidml_cc_library(
    name = "support",
    srcs = ["support.cc"],
    hdrs = ["support.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":proto_cc",
        "//tile/base",
        "//tile/lang",
    ],
)
