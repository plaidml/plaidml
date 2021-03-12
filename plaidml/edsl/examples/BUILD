# Copyright 2020 Intel Corporation

load("@rules_python//python:defs.bzl", "py_test")
load("//bzl:plaidml.bzl", "plaidml_cc_test")

filegroup(
    name = "docs",
    srcs = [
        "complex_conv_2d.cc",
        "complex_conv_2d.py",
        "conv_1d.cc",
        "conv_1d.py",
        "conv_2d_dilated.cc",
        "conv_2d_dilated.py",
        "gemm.cc",
        "gemm.py",
        "gemv.cc",
        "gemv.py",
        "max_pool_1d.cc",
        "max_pool_1d.py",
        "quantize.cc",
        "quantize.py"
    ],
    visibility = ["//visibility:public"],
)

plaidml_cc_test(
    name = "examples_cc_test",
    srcs = ["complex_conv_2d.cc",
            "conv_1d.cc",
            "conv_2d_dilated.cc",
            "gemm.cc",
            "gemv.cc",
            "max_pool_1d.cc",
            "quantize.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml:testenv",
        "//plaidml/edsl",
        "//plaidml/exec",
    ],
)

py_test(
    name = "examples_py_test",
    size = "medium",
    srcs = ["complex_conv_2d.py",
            "conv_1d.py",
            "conv_2d_dilated.py",
            "gemm.py",
            "gemv.py",
            "max_pool_1d.py",
            "quantize.py",
            "example_tests.py"],
    args = ["-v"],
    main = "example_tests.py",
    deps = [
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
)
