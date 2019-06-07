# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_jsonnet//jsonnet:jsonnet.bzl", "jsonnet_to_json")
load("//bzl:plaidml.bzl", "plaidml_cc_library")

jsonnet_to_json(
    name = "cpu_jsonnet",
    src = "cpu.jsonnet",
    outs = ["cpu.json"],
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
        "//tile/stripe",
        "@half",
        "@llvm_shim//:llvm",
    ],
    alwayslink = 1,
)

plaidml_cc_library(
    name = "jit",
    srcs = glob([
        "jit.cc",
        "jit.h",
    ]),
    copts = [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
    ],
    tags = ["llvm"],
    deps = [
        "//tile/stripe",
        "@half",
        "@llvm_shim//:llvm",
    ],
)
