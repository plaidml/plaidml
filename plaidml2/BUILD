# Copyright 2019 Intel Corporation.

load("//tools/py_cffi:build_defs.bzl", "py_cffi")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_dylib",
    "plaidml_cc_library",
    "plaidml_py_library",
    "plaidml_py_wheel",
)

plaidml_cc_library(
    name = "plaidml",
    deps = [
        "//plaidml2/core",
        "//plaidml2/edsl",
        "//plaidml2/exec",
        "//plaidml2/op",
    ],
)

plaidml_cc_dylib(
    name = "dylib",
    dylib_name = "plaidml",
    visibility = ["//visibility:public"],
    deps = [
        ":plaidml",
    ],
)

plaidml_cc_library(
    name = "api",
    srcs = [":dylib"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml2/core:api",
        "//plaidml2/edsl:api",
        "//plaidml2/exec:api",
        "//plaidml2/op:api",
    ],
)

plaidml_py_library(
    name = "py",
    srcs = [
        "__init__.py",
        "ffi.py",
        "settings.py",
    ],
    data = [":dylib"],
    visibility = ["//visibility:public"],
    deps = [
        ":_ffi",
    ],
)

py_cffi(
    name = "_ffi",
    module = "plaidml2._ffi",
    srcs_ordered = [
        "//plaidml2/core:ffi.h",
        "//plaidml2/edsl:ffi.h",
        "//plaidml2/exec:ffi.h",
        "//plaidml2/op:ffi.h",
    ],
)
