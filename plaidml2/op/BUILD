# Copyright 2019 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_test")
load("@rules_pkg//:pkg.bzl", "pkg_tar")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
    "plaidml_py_library",
)

exports_files([
    "ffi.h",
])

pkg_tar(
    name = "include_pkg",
    srcs = [
        "ffi.h",
        "op.h",
    ],
    package_dir = "include/plaidml2/op",
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
    defines = ["PLAIDML_OP_DLL"],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//plaidml2/core",
        "//plaidml2/edsl",
        "//plaidml2/op/lib",
        "@boost",
    ],
)

plaidml_cc_library(
    name = "api",
    hdrs = [
        "ffi.h",
        "op.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml2/core:api",
        "//plaidml2/edsl:api",
    ],
)

plaidml_py_library(
    name = "py",
    srcs = [
        "__init__.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml2:py",
        "//plaidml2/edsl:py",
    ],
)

plaidml_cc_test(
    name = "cc_test",
    srcs = ["op_test.cc"],
    deps = [
        ":api",
        ":op",
    ],
)

py_test(
    name = "py_test",
    srcs = ["op_test.py"],
    main = "op_test.py",
    deps = [":py"],
)
