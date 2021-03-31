# Copyright 2019, Intel Corporation.

load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

# NOTE: We can't statically link against anything unless it was built with _GLIBCXX_USE_CXX11_ABI=0.
#       This is because the pytorch libs were built with an older gcc (pre-C++ 11 ABI).

plaidml_cc_library(
    name = "pytorch",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    tags = [
        "manual",
        "pytorch",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml:lib",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": ["@com_intel_plaidml_conda//:pytorch"],
    }),
    alwayslink = 1,
)

plaidml_cc_binary(
    name = "plaidml_pytorch.so",
    linkshared = 1,
    tags = [
        "manual",
        "pytorch",
    ],
    deps = [":pytorch"],
)

py_library(
    name = "py",
    srcs = ["__init__.py"],
    data = [":plaidml_pytorch.so"],
    tags = [
        "manual",
        "pytorch",
    ],
    visibility = ["//visibility:public"],
    deps = ["//plaidml:py"],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    tags = [
        "manual",
        "pytorch",
    ],
    deps = [":py"],
)

py_test(
    name = "ops_test",
    srcs = ["ops_test.py"],
    tags = [
        "manual",
        "pytorch",
    ],
    deps = [":test_utils"],
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    tags = [
        "manual",
        "pytorch",
    ],
    deps = [":test_utils"],
)
