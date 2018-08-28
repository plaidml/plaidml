# Copyright 2017, Vertex.AI.
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
        "//tile/proto:metadata_cc",
    ],
)
