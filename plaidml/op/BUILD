# Copyright 2019 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_test")
load("@rules_pkg//:pkg.bzl", "pkg_tar")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

SDK_HDRS = [
    "ffi.h",
    "op.h",
]

exports_files([
    "ffi.h",
])

pkg_tar(
    name = "include_pkg",
    srcs = SDK_HDRS,
    package_dir = "include/plaidml/op",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "sdk",
    srcs = SDK_HDRS,
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "op",
    srcs = [
        "ffi.cc",
    ],
    hdrs = [
        "ffi.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml/core",
        "//plaidml/edsl",
        "//plaidml/op/lib",
        "//pmlc/util",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

plaidml_cc_library(
    name = "api",
    hdrs = SDK_HDRS,
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml/core:api",
        "//plaidml/edsl:api",
    ],
)

py_library(
    name = "py",
    srcs = [
        "__init__.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml:py",
        "//plaidml/edsl:py",
    ],
)

plaidml_cc_test(
    name = "cc_test",
    srcs = ["op_test.cc"],
    deps = [
        ":api",
        ":op",
        "//plaidml:testenv",
    ],
)

py_test(
    name = "py_test",
    srcs = ["op_test.py"],
    main = "op_test.py",
    deps = [":py"],
)
