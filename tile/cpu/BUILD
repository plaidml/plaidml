# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_binary")

plaidml_cc_binary(
    name = "cpu",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
    ),
    copts = [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
    ],
    tags = ["llvm"],
    visibility = ["//visibility:public"],
    deps = [
        "//base/config",
        "//tile/bilp",
        "//tile/codegen",
        "//tile/lang",
        "//tile/lib",
        "//tile/stripe",
        "//tile/targets",
    ],
)
