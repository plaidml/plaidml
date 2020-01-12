# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

SDK_HDRS = [
    "exec.h",
    "ffi.h",
]

exports_files([
    "ffi.h",
])

filegroup(
    name = "sdk",
    srcs = SDK_HDRS,
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "exec",
    srcs = [
        "ffi.cc",
    ],
    hdrs = SDK_HDRS,
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml/core",
        "//pmlc/compiler",
        "//pmlc/target/intel_gen",
        "//pmlc/target/x86",
        "//pmlc/util",
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
    ],
)
