# Copyright 2019 Intel Corporation.

load("@rules_pkg//:pkg.bzl", "pkg_tar")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
    "plaidml_cc_version",
    "plaidml_py_version",
)

SDK_HDRS = [
    "core.h",
    "ffi.h",
]

exports_files([
    "ffi.h",
])

pkg_tar(
    name = "include_pkg",
    srcs = SDK_HDRS,
    package_dir = "include/plaidml2/core",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "sdk",
    srcs = SDK_HDRS,
    visibility = ["//visibility:public"],
)

plaidml_cc_version(
    name = "cc_version",
    prefix = "PLAIDML",
)

plaidml_py_version(
    name = "py_version",
    prefix = "PLAIDML",
)

plaidml_cc_library(
    name = "core",
    srcs = [
        "ffi.cc",
        "internal.h",
        "settings.cc",
        "settings.h",
        ":cc_version",
    ],
    hdrs = SDK_HDRS,
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/dialect/tile",
        "//pmlc/util",
        "@boost//:filesystem",
        "@boost//:thread",
    ],
    alwayslink = 1,
)

plaidml_cc_library(
    name = "api",
    hdrs = SDK_HDRS,
    visibility = ["//visibility:public"],
)

py_library(
    name = "py",
    srcs = [
        "__init__.py",
        "settings.py",
        ":py_version",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "py_test",
    srcs = ["core_test.py"],
    main = "core_test.py",
    deps = [
        ":py",
        "//plaidml2:py",
    ],
)
