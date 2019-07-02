# Copyright 2019 Intel Corporation.

load("//bzl:conda.bzl", "conda_test")
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
    name = "edsl",
    srcs = [
        "ffi.cc",
    ],
    hdrs = [
        "edsl.h",
        "ffi.h",
    ],
    defines = ["PLAIDML_EDSL_DLL"],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//plaidml2/core",
        "//tile/lang",
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
    hdrs = [
        "edsl.h",
        "ffi.h",
    ],
    visibility = ["//visibility:public"],
    deps = ["//plaidml2/core:api"],
)

plaidml_cc_test(
    name = "cc_test",
    srcs = ["edsl_test.cc"],
    deps = [
        ":api",
        ":edsl",
    ],
)

conda_test(
    name = "py_test",
    srcs = ["edsl_test.py"],
    env = "@com_intel_plaidml_conda//:env",
    main = "edsl_test.py",
    deps = [":py"],
)
