# Copyright 2019 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_library")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

SDK_HDRS = [
    "ffi.h",
    "op.h",
]

exports_files([
    "ffi.h",
])

filegroup(
    name = "sdk",
    srcs = SDK_HDRS,
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "sdk_tar",
    srcs = [":sdk"],
    package_dir = "include/plaidml/op",
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "headers",
    hdrs = [
        "ffi.h",
        "op.h",
    ],
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "op",
    srcs = [
        "ffi.cc",
    ],
    hdrs = [
        "ffi.h",
        "op.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml/core",
        "//plaidml/edsl",
        "//plaidml/op/lib",
        "//pmlc/util",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
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
    name = "op_test",
    size = "medium",
    srcs = ["op_test.cc"],
    linkstatic = True,
    deps = [
        ":op",
        "//plaidml:testenv",
    ],
)
