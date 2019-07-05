# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "lib",
    srcs = [
        "ops.cc",
    ],
    hdrs = [
        "ops.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//plaidml2/core",
        "//plaidml2/edsl",
        "@boost",
    ],
)
