# Copyright 2019 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_test")
load("@rules_pkg//:pkg.bzl", "pkg_tar")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
    "plaidml_py_library",
)

SDK_HDRS = [
    "ffi.h",
    "op.h",
]

exports_files([
    "ffi.h",
])

pkg_tar(
    name = "include_pkg",
    srcs = SDK_HDRS,
    package_dir = "include/plaidml2/op",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "sdk",
    srcs = SDK_HDRS,
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "op_ast",
    srcs = [
        "ffi.cc",
    ],
    hdrs = [
        "ffi.h",
    ],
    defines = [
        "PLAIDML_OP_DLL",
        "PLAIDML_AST",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//plaidml2/core:core_ast",
        "//plaidml2/edsl:edsl_ast",
        "//plaidml2/op/lib:lib_ast",
        "@boost",
    ],
    alwayslink = 1,
)

plaidml_cc_library(
    name = "op_mlir",
    srcs = [
        "ffi.cc",
    ],
    hdrs = [
        "ffi.h",
    ],
    defines = [
        "PLAIDML_OP_DLL",
        "PLAIDML_MLIR",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//plaidml2/core:core_mlir",
        "//plaidml2/edsl:edsl_mlir",
        "//plaidml2/op/lib:lib_mlir",
        "@boost",
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
        "//plaidml2/edsl:py",
    ],
)

plaidml_cc_test(
    name = "cc_test_ast",
    srcs = ["op_test.cc"],
    deps = [
        ":api",
        ":op_ast",
        "//plaidml2:testenv_ast",
    ],
)

plaidml_cc_test(
    name = "cc_test_mlir",
    srcs = ["op_test.cc"],
    deps = [
        ":api",
        ":op_mlir",
        "//plaidml2:testenv_mlir",
    ],
)

py_test(
    name = "py_test",
    srcs = ["op_test.py"],
    main = "op_test.py",
    deps = [":py"],
)
