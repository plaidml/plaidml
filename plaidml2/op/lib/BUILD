# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "lib_ast",
    srcs = [
        "ops.cc",
    ],
    hdrs = [
        "ops.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//plaidml2/core:core_ast",
        "//plaidml2/edsl:edsl_ast",
        "//plaidml2/op:api",
        "@boost",
    ],
)

plaidml_cc_library(
    name = "lib_mlir",
    srcs = [
        "ops.cc",
    ],
    hdrs = [
        "ops.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//plaidml2/core:core_mlir",
        "//plaidml2/edsl:edsl_mlir",
        "//plaidml2/op:api",
        "@boost",
    ],
)
