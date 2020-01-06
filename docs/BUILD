load("@rules_python//python:defs.bzl", "py_binary")

filegroup(
    name = "docs",
    srcs = glob([
        "Doxyfile",
        "conf.py",
        "**/*.rst",
    ]),
    visibility = ["//visibility:public"],
)

py_binary(
    name = "watch",
    srcs = [
        "watch.py",
    ],
    data = [
        ":docs",
        "//plaidml2/edsl:docs",
    ],
    deps = [
        "//plaidml2:py",
        "//plaidml2/edsl:py",
        "//plaidml2/exec:py",
        "//plaidml2/op:py",
        "//tools/py_setup",
    ],
)
