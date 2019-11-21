# Copyright 2018 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_binary")

package(default_visibility = ["//visibility:public"])

plaidml_cc_binary(
    name = "resnet50",
    srcs = ["resnet50.cc"],
    deps = [
        "//base/util",
        "//plaidml2/edsl:api",
        "//plaidml2/edsl:edsl_ast",
        "//plaidml2/exec:api",
        "//plaidml2/exec:exec_ast",
        "//plaidml2/op:api",
        "//plaidml2/op:op_ast",
    ],
)
