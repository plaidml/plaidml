# Copyright 2019 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_test")
load("@rules_pkg//:pkg.bzl", "pkg_tar")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

SDK_HDRS = [
    "autodiff.h",
    "edsl.h",
    "ffi.h",
]

exports_files([
    "ffi.h",
])

filegroup(
    name = "docs",
    srcs = [
        "edsl.h",
        "edsl_doc.cc",
        "edsl_doc.py",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "sdk",
    srcs = SDK_HDRS,
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "edsl",
    srcs = [
        "derivs.cc",
        "derivs.h",
        "ffi.cc",
    ],
    hdrs = SDK_HDRS,
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml/core",
        "//pmlc/dialect/tile",
        "//pmlc/util",
        "@llvm-project//llvm:support",
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
    ],
)

plaidml_cc_test(
    name = "cc_test",
    srcs = ["edsl_test.cc"],
    deps = [
        ":edsl",
        "//plaidml:testenv",
        "//plaidml/exec",
    ],
)

py_test(
    name = "py_test",
    srcs = ["edsl_test.py"],
    args = ["-v"],
    main = "edsl_test.py",
    deps = [
        ":py",
        "//plaidml/exec:py",
    ],
)

plaidml_cc_test(
    name = "cc_doc",
    srcs = ["edsl_doc.cc"],
    deps = [
        ":edsl",
        "//plaidml:testenv",
        "//plaidml/exec",
    ],
)

py_test(
    name = "py_doc",
    srcs = ["edsl_doc.py"],
    main = "edsl_doc.py",
    deps = [
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
)
