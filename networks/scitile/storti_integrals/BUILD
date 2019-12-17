# Copyright 2018 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "torus",
    srcs = ["torus.py",
            "op.py",],
    deps = [
        "//plaidml2:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
    ],
)

py_binary(
    name="op",
    srcs = ["op.py"],
    deps =[
        "//plaidml2:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
    ],
)
py_binary(
    name = "hypersphere",
    srcs = ["hypersphere.py",
            "op.py",],
    deps = [
        "//plaidml2:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
    ],
)

py_test(
    name = "torus_tests",
    srcs = [
        "torus_tests.py",
        "torus.py",
    ],
    deps = [
        "//plaidml2:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
    ],
)

