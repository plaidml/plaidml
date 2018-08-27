load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("@vertexai_plaidml//bzl:conda.bzl", "conda_test")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_py_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "matchers",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        "//base/util",
        "//plaidml/base",
        "@gmock_archive//:gtest",
    ],
)

plaidml_cc_library(
    name = "plaidml_config",
    testonly = True,
    srcs = ["plaidml_config.cc"],
    hdrs = ["plaidml_config.h"],
    data = ["//plaidml:configs"],
    deps = [
        "//base/util",
        "//base/util:runfiles_db",
    ],
)

plaidml_py_library(
    name = "plaidml_py",
    testonly = True,
    srcs = [
        "__init__.py",
        "plaidml_config.py",
    ],
    data = ["//plaidml:configs"],
)

conda_test(
    name = "configs_test",
    srcs = ["configs_test.py"],
    data = ["//plaidml:configs"],
    env = "@vertexai_plaidml_conda//:env",
    deps = ["//plaidml:py"],
)

plaidml_cc_library(
    name = "gtest_main",
    testonly = True,
    srcs = ["gtest_main.cc"],
    deps = [
        "//base/util",
        "@com_github_gflags_gflags//:gflags",
        "@easylogging_repo//:easylogging",
        "@gmock_archive//:gtest",
    ],
)
