# Copyright 2019 Intel Corporation.

load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
    "plaidml_py_library",
)

exports_files([
    "ffi.h",
])

plaidml_cc_library(
    name = "exec",
    srcs = [
        "ffi.cc",
    ],
    hdrs = [
        "exec.h",
        "ffi.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//plaidml2/core",
        "//tile/hal/cpu",
        "//tile/hal/opencl",
        "//tile/lang",
        "//tile/platform/local_machine",
        "//tile/targets",
    ] + select({
        "@toolchain//:macos_x86_64": [
            "//tile/hal/metal",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

plaidml_cc_library(
    name = "api",
    hdrs = [
        "exec.h",
        "ffi.h",
    ],
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
