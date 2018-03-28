# Copyright Vertex.AI.

licenses(["by_exception_only"])

load("//bzl:plaidml.bzl", "plaidml_py_library", "plaidml_py_test", "plaidml_py_wheel")
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_deb", "pkg_tar")

package(default_visibility = ["//visibility:public"])

exports_files(["requirements.txt"])

plaidml_py_library(
    name = "keras",
    srcs = [
        "__init__.py",
        "backend.py",
    ],
    deps = [
        "//plaidml:py",
    ],
)

pkg_tar(
    name = "pkg",
    srcs = glob([
        "**/*",
    ]),
    package_dir = "plaidml/keras",
    strip_prefix = ".",
)

plaidml_py_test(
    name = "backend_test",
    timeout = "long",
    srcs = [
        "backend_test.py",
    ],
    requirements = "backend_test.txt",
    tags = [
        "keras",
        "manual",
        "tensorflow",
    ],
    deps = [
        ":keras",
        "//testing:plaidml_py",
    ],
)

plaidml_py_test(
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
        "//testing:plaidml_py",
    ],
)

plaidml_py_test(
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
        "//testing:plaidml_py",
    ],
)

plaidml_py_test(
    name = "tile_sandbox",
    srcs = [
        "tile_sandbox.py",
    ],
    tags = [
        "keras",
        "manual",
    ],
    deps = [
        ":keras",
        "//testing:plaidml_py",
    ],
)

plaidml_py_test(
    name = "trivial_model_test",
    srcs = [
        "trivial_model_test.py",
    ],
    tags = [
        "keras",
        "manual",
    ],
    deps = [
        ":keras",
        "//testing:plaidml_py",
    ],
)

plaidml_py_test(
    name = "tutorial_test",
    srcs = [
        "tutorial_test.py",
    ],
    tags = [
        "keras",
        "trusty",
    ],
    deps = [
        ":keras",
        "//testing:plaidml_py",
    ],
)

plaidml_py_wheel(
    name = "wheel",
    package_name = "plaidml_keras",
    srcs = [
        "__init__.py",
        "backend.py",
    ],
    config = ":setup.cfg",
    package_prefix = "plaidml/keras",
    python = "py2.py3",
)
