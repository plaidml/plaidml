# Copyright 2019 Intel Corporation.

load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
)

plaidml_cc_library(
    name = "ast",
    srcs = [
        "ast.cc",
        "ast_ops.cc",
        "ast_ops.h",
        "gradient.cc",
    ],
    hdrs = [
        "ast.h",
        "gradient.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//tile/base",
        "//tile/lang",
        "@boost",
    ],
    alwayslink = 1,
)
