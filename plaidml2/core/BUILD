# Copyright 2019 Intel Corporation.

load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
    "plaidml_py_library",
)

exports_files([
    "ffi.h",
])

plaidml_cc_library(
    name = "core",
    srcs = [
        "ffi.cc",
        "internal.h",
    ],
    hdrs = [
        "core.h",
        "ffi.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//tile/lang",
        "@boost//:filesystem",
    ],
    alwayslink = 1,
)

plaidml_cc_library(
    name = "api",
    hdrs = [
        "core.h",
        "ffi.h",
    ],
    visibility = ["//visibility:public"],
)

plaidml_py_library(
    name = "py",
    srcs = [
        "__init__.py",
    ],
    visibility = ["//visibility:public"],
)
