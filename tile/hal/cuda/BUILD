# Copyright 2018, Intel Corporation.

load("@cuda//:build_defs.bzl", "if_cuda_is_configured")
load(
    "//bzl:plaidml.bzl",
    "plaidml_proto_library",
    "plaidml_cc_library",
    "plaidml_cc_test",
)

plaidml_cc_library(
    name = "cuda",
    srcs = if_cuda_is_configured([
        "emit.cc",
        "error.h",
        "hal.cc",
        "compiler.cc",
    ]),
    hdrs = [
        "emit.h",
        "hal.h",
    ],
    tags = ["cuda"],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml:proto_cc",
        "//tile/base:hal",
    ] + if_cuda_is_configured([
        "@cuda//:cuda_headers",
        "@cuda//:cuda_driver",
        "@cuda//:nvrtc",
    ]),
    alwayslink = 1,
)

plaidml_cc_test(
    name = "platform_test",
    srcs = if_cuda_is_configured(["platform_test.cc"]),
    tags = ["cuda"],
    deps = [
        ":cuda",
        "//tile/base:platform_test",
        "//tile/platform/local_machine",
    ],
)
