load("@rules_python//python:defs.bzl", "py_binary")

filegroup(
    name = "runfiles",
    srcs = glob([
        "Doxyfile",
        "conf.py",
        "**/*.rst",
    ]),
    visibility = ["//visibility:public"],
)

py_binary(
    name = "docs",
    srcs = [
        "docs.py",
    ],
    data = [
        ":runfiles",
        "//plaidml/core:docs",
        "//plaidml/edsl:docs",
        "//plaidml/exec:docs",
    ],
    deps = [
        "//plaidml:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
        "//plaidml/op:py",
        "//tools/py_setup",
    ],
)
