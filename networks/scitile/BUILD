# Copyright 2018 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "uw_toroidal_shell",
    srcs = ["uw_toroidal_shell.py"],
    deps = [
        "//plaidml2/bridge/keras",
    ],
)

py_test(
    name = "uw_tests",
    srcs = [
        "uw_tests.py",
        "uw_toroidal_shell.py"
    ],
    tags = [
        "keras",
        "manual",
    ],
  deps = [
        "//plaidml2/bridge/keras",
    ],
)