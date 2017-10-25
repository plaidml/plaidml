# Copyright Vertex.AI.

licenses(["by_exception_only"])

load("//bzl:plaidml.bzl", "plaidml_py_library", "plaidml_py_test", "plaidml_py_wheel")

package(default_visibility = ["//visibility:public"])

plaidml_py_library(
    name = "keras",
    srcs = [
        "__init__.py",
        "backend.py",
    ],
    deps = [
        "//plaidml:py",
        "//plaidml/event:filelog_py",
    ],
)

plaidml_py_test(
    name = "backend_test",
    timeout = "long",
    srcs = [
        "backend_test.py",
    ],
    tags = [
        "darwin_fail",
        "keras",
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
    name = "trivial_model_test",
    srcs = [
        "trivial_model_test.py",
    ],
    tags = [
        "keras",
    ],
    deps = [
        ":keras",
        "//testing:plaidml_py",
    ],
)

plaidml_py_wheel(
    name = "wheel",
    srcs = [
        "__init__.py",
        "backend.py",
    ],
    config = ":setup.cfg",
    package = "plaidml/keras",
    python = "py2.py3",
-