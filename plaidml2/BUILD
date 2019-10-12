# Copyright 2019 Intel Corporation.

load("//tools/py_cffi:build_defs.bzl", "py_cffi")
load("//tools/py_setup:build_defs.bzl", "py_setup")
load("@rules_python//python:defs.bzl", "py_binary")
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
    dylib_name = "plaidml2",
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
    ],
    data = [":dylib"],
    visibility = ["//visibility:public"],
    deps = [
        ":_ffi",
        "//plaidml2/core:py",
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

py_binary(
    name = "setup",
    srcs = ["plaidml_setup.py"],
    main = "plaidml_setup.py",
    deps = [
        ":py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
        "//plaidml2/op:py",
    ],
)

py_binary(
    name = "py_setup",
    srcs = ["setup.py"],
    data = [
        ":api",
        "//plaidml2/core:sdk",
        "//plaidml2/edsl:sdk",
        "//plaidml2/exec:sdk",
        "//plaidml2/op:sdk",
    ],
    main = "setup.py",
    deps = [
        ":py",
        "//plaidml2/core:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
        "//plaidml2/op:py",
        "//tools/py_setup",
    ],
)

py_setup(
    name = "wheel",
    package_name = "plaidml2",
    platform = select({
        "//toolchain:macos_x86_64": "macosx_10_10_x86_64",
        "//toolchain:windows_x86_64": "win_amd64",
        "//conditions:default": "manylinux1_x86_64",
    }),
    tool = ":py_setup",
    universal = True,
    visibility = ["//visibility:public"],
)
