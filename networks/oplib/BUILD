# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "oplib",
    srcs = ["resnet50.cc"],
    hdrs = ["oplib.h"],
    deps = [
        "//plaidml:lib",
        "//pmlc/util",
        "@llvm-project//llvm:Support",
    ],
)

plaidml_cc_binary(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    deps = [
        ":oplib",
        "//plaidml/exec",
        "//pmlc/rt",
        "//pmlc/util",
        "@com_github_google_benchmark//:benchmark",
    ],
)

plaidml_cc_binary(
    name = "run",
    srcs = ["run.cc"],
    deps = [
        ":oplib",
        "//plaidml/exec",
        "//pmlc/rt",
        "//pmlc/util",
    ],
)
