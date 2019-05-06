# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

filegroup(
    name = "configs",
    srcs = glob(["*.jsonnet"]),
)

plaidml_cc_library(
    name = "cpu",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    copts = [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
    ],
    tags = ["llvm"],
    deps = [
        "//tile/codegen",
        "//tile/stripe",
        "@half",
        "@llvm_shim//:llvm",
    ],
    alwayslink = 1,
)
