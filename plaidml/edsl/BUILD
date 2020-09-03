# Copyright 2020 Intel Corporation

load("@rules_python//python:defs.bzl", "py_library", "py_test")
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
    srcs = ["edsl_docs.py"],
    main = "edsl_docs.py",
    deps = [
        ":py",
        "//plaidml/exec:py",
    ],
)
