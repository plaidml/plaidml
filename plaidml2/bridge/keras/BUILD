# Copyright 2018 Intel Corporation.

licenses(["by_exception_only"])

load("@rules_python//python:defs.bzl", "py_test")
load("//tools/py_setup:build_defs.bzl", "py_setup")
load("//bzl:plaidml.bzl", "plaidml_py_library")

package(default_visibility = ["//visibility:public"])

plaidml_py_library(
    name = "keras",
    srcs = [
        "__init__.py",
    ],
    deps = [
        "//plaidml2:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
        "//plaidml2/op:py",
    ],
)

py_binary(
    name = "setup",
    srcs = ["setup.py"],
    deps = [
        ":keras",
        "//tools/py_setup",
    ],
)

py_setup(
    name = "wheel",
    package_name = "plaidml2_keras",
    tool = ":setup",
    universal = True,
)

py_test(
    name = "backend_test",
    timeout = "long",
    srcs = [
        "backend_test.py",
    ],
    tags = [
        "keras",
        "manual",
        "tensorflow",
    ],
    deps = [
        ":keras",
    ],
)

py_test(
    name = "mnist_mlp_test",
    srcs = [
        "mnist_mlp_test.py",
    ],
    tags = [
        "keras",
        "manual",
    ],
    deps = [
        ":keras",
    ],
)

py_test(
    name = "regression_test",
    srcs = [
        "regression_test.py",
    ],
    tags = [
        "keras",
        "manual",
    ],
    deps = [
        ":keras",
    ],
)

# py_test(
#     name = "tf_test",
#     srcs = [
#         "tf_test.py",
#     ],
#     tags = [
#         "keras",
#         "tensorflow",
#     ],
#     deps = [
#         ":keras",
#     ],
# )

# py_test(
#     name = "tile_sandbox",
#     srcs = [
#         "tile_sandbox.py",
#     ],
#     tags = [
#         "keras",
#         "manual",
#     ],
#     deps = [
#         ":keras",
#     ],
# )

py_test(
    name = "trivial_model_test",
    srcs = ["trivial_model_test.py"],
    tags = [
        "keras",
        "manual",
    ],
    deps = [
        ":keras",
    ],
)

# py_test(
#     name = "tutorial_test",
#     srcs = [
#         "tutorial_test.py",
#     ],
#     tags = [
#         "keras",
#     ],
#     deps = [
#         ":keras",
#     ],
# )
