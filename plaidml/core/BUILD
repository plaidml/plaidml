# Copyright 2020 Intel Corporation

load("@rules_python//python:defs.bzl", "py_library", "py_test")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_version",
    "plaidml_py_version",
)
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

SDK_HDRS = [
    "core.h",
    "ffi.h",
]

exports_files([
    "ffi.h",
])

filegroup(
    name = "docs",
    srcs = ["core.h"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "sdk",
    srcs = SDK_HDRS,
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "sdk_tar",
    srcs = [":sdk"],
    package_dir = "include/plaidml/core",
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
        "//pmlc/ast",
        "//pmlc/compiler",
        "//pmlc/util",
        "@llvm-project//llvm:Support",
    ],
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
    size = "small",
    srcs = ["core_test.py"],
    args = ["-v"],
    main = "core_test.py",
    deps = [
        ":py",
        "//plaidml:py",
        "//plaidml/edsl:py",
    ],
)
