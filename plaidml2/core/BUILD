# Copyright 2019 Intel Corporation.

load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
    "plaidml_cc_version",
    "plaidml_py_library",
    "plaidml_py_version",
)

exports_files([
    "ffi.h",
])

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
    hdrs = [
        "core.h",
        "ffi.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//tile/lang/ast",
        "@boost//:filesystem",
    ],
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
        "settings.py",
        ":py_version",
    ],
    visibility = ["//visibility:public"],
)
