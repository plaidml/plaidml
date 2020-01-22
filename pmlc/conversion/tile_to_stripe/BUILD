# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "tile_to_stripe",
    srcs = [
        "tile_to_stripe.cc",
    ],
    hdrs = [
        "tile_to_stripe.h",
    ],
    deps = [
        "//base/util",
        "//pmlc/dialect/eltwise",
        "//pmlc/dialect/stripe:transcode",
        "//pmlc/dialect/tile/ir",
        "//pmlc/dialect/tile/transforms",
        "//pmlc/util",
        "//tile/base",
    ],
    alwayslink = 1,
)
