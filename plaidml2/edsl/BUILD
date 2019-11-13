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
    "autodiff.h",
    "edsl.h",
    "ffi.h",
]

exports_files([
    "ffi.h",
])

pkg_tar(
    name = "include_pkg",
    srcs = SDK_HDRS,
    package_dir = "include/plaidml2/edsl",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "sdk",
    srcs = SDK_HDRS,
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "edsl_ast",
    srcs = [
        "derivs.cc",
        "derivs.h",
        "ffi.cc",
    ],
    hdrs = SDK_HDRS,
    defines = [
        "PLAIDML_EDSL_DLL",
        "PLAIDML_AST",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//plaidml2/core:core_ast",
        "//tile/lang/ast",
    ],
    alwayslink = 1,
)

plaidml_cc_library(
    name = "edsl_mlir",
    srcs = [
        "derivs.cc",
        "derivs.h",
        "ffi.cc",
    ],
    hdrs = SDK_HDRS,
    defines = [
        "PLAIDML_EDSL_DLL",
        "PLAIDML_MLIR",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//plaidml2/core:core_mlir",
        "//pmlc/dialect/tile",
    ],
    alwayslink = 1,
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

plaidml_cc_library(
    name = "api",
    hdrs = SDK_HDRS,
    visibility = ["//visibility:public"],
    deps = ["//plaidml2/core:api"],
)

plaidml_cc_test(
    name = "cc_test_ast",
    srcs = ["edsl_test.cc"],
    deps = [
        ":api",
        ":edsl_ast",
        "//plaidml2:testenv_ast",
        "//plaidml2/exec:exec_ast",
    ],
)

plaidml_cc_test(
    name = "cc_test_mlir",
    srcs = ["edsl_test.cc"],
    deps = [
        ":api",
        ":edsl_mlir",
        "//plaidml2:testenv_mlir",
        "//plaidml2/exec:exec_mlir",
    ],
)

py_test(
    name = "py_test",
    srcs = ["edsl_test.py"],
    main = "edsl_test.py",
    deps = [
        ":py",
        "//plaidml2/exec:py",
    ],
)
