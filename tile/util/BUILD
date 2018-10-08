# Copyright 2017-2018 Intel Corporation.
load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "util",
    srcs = [
        "tile_file.cc",
    ],
    hdrs = [
        "tile_file.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//tile/lang",
        "//tile/proto:metadata_cc",
        "//tile/proto:proto_cc",
    ],
)
