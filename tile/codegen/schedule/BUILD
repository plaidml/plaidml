# Copyright 2018, Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "schedule",
    srcs = [
        "deps.cc",
        "deps.h",
        "placer.cc",
        "placer.h",
    ],
    hdrs = [
        "scheduler.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tile/stripe",
        "//tile/codegen",
    ],
)

plaidml_cc_test(
    name = "deps_test",
    srcs = ["deps_test.cc"],
    deps = [
        ":schedule",
        "//testing:matchers",
    ],
)

plaidml_cc_test(
    name = "placer_test",
    srcs = ["placer_test.cc"],
    deps = [
        ":schedule",
        "//testing:matchers",
    ],
)
