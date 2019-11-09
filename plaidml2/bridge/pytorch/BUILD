# Copyright 2019, Intel Corporation.

load("@rules_python//python:defs.bzl", "py_test")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_binary",
    "plaidml_cc_library",
    "plaidml_py_library",
)

# NOTE: We can't statically link against anything unless it was built with _GLIBCXX_USE_CXX11_ABI=0.
#       This is because the pytorch libs were built with an older gcc (pre-C++ 11 ABI).

plaidml_cc_library(
    name = "pytorch",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    tags = ["pytorch"],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml2:api_ast",
        "@com_intel_plaidml_conda_unix//:pytorch",
    ],
    alwayslink = 1,
)

plaidml_cc_binary(
    name = "plaidml_pytorch.so",
    linkshared = 1,
    tags = ["pytorch"],
    deps = [":pytorch"],
)

plaidml_py_library(
    name = "py",
    srcs = ["__init__.py"],
    data = [":plaidml_pytorch.so"],
    tags = ["pytorch"],
    visibility = ["//visibility:public"],
    deps = ["//plaidml2:py"],
)

plaidml_py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    tags = ["pytorch"],
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
