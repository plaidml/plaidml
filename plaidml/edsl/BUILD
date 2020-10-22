# Copyright 2020 Intel Corporation

load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

SDK_HDRS = [
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
        "edsl_docs.cc",
        "edsl_docs.py",
    ],
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
    package_dir = "include/plaidml/edsl",
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "edsl",
    srcs = ["ffi.cc"],
    hdrs = SDK_HDRS,
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml/core",
        "//pmlc/ast",
        "//pmlc/util",
        "@llvm-project//llvm:Support",
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
    ],
)

plaidml_cc_test(
    name = "cc_docs",
    size = "small",
    srcs = ["edsl_docs.cc"],
    linkstatic = True,
    deps = [
        ":edsl",
        "//plaidml:testenv",
        "//plaidml/exec",
    ],
)

py_test(
    name = "py_docs",
    size = "small",
    srcs = ["edsl_docs.py"],
    main = "edsl_docs.py",
    deps = [
        ":py",
        "//plaidml/exec:py",
    ],
)
