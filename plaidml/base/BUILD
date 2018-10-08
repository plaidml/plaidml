# Copyright 2018 Intel Corporation.
package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test", "plaidml_py_library", "plaidml_py_init")
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

plaidml_cc_library(
    name = "base",
    srcs = [
        "context.cc",
        "feature.cc",
        "logging.cc",
        "perf_counter.cc",
        "status.cc",
        "status_strings.cc",
    ],
    hdrs = [
        "base.h",
        "base_cpp.h",
        "context.h",
        "status.h",
        "status_strings.h",
    ],
    defines = ["VAI_DLL"],
    deps = [
        "//base/config",
        "//base/context",
        "//base/util",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "status_test",
    srcs = ["status_test.cc"],
    deps = [
        ":base",
    ],
)

plaidml_cc_test(
    name = "feature_test",
    srcs = ["feature_test.cc"],
    deps = [
        ":base",
    ],
)

pkg_tar(
    name = "sdk_includes",
    srcs = [
        "base.h",
        "base_cpp.h",
    ],
    package_dir = "include/plaidml/base",
)
