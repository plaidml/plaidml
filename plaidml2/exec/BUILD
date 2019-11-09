# Copyright 2019 Intel Corporation.

load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
    "plaidml_py_library",
)

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
    name = "exec_ast",
    srcs = [
        "ffi.cc",
    ],
    hdrs = SDK_HDRS,
    defines = [
        "PLAIDML_EXEC_DLL",
        "PLAIDML_AST",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//plaidml2/core:core_ast",
    ],
    alwayslink = 1,
)

plaidml_cc_library(
    name = "exec_mlir",
    srcs = [
        "ffi.cc",
    ],
    hdrs = SDK_HDRS,
    defines = [
        "PLAIDML_EXEC_DLL",
        "PLAIDML_MLIR",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//plaidml2/core:core_mlir",
        "//pmlc/dialect/stripe:transcode",
    ],
    alwayslink = 1,
)

plaidml_cc_library(
    name = "api",
    hdrs = SDK_HDRS,
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
    ],
)
