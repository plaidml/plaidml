# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary")

plaidml_cc_binary(
    name = "gencfg",
    srcs = ["gencfg.cc"],
    deps = [
        "//base/config",
        "//base/util",
        "//tile/codegen:proto_cc",
        "@boost//:program_options",
    ],
)
