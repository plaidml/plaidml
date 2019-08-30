package(default_visibility = ["//visibility:public"])

load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "report",
    srcs = [
        "report.py",
        "util.py",
    ],
    data = glob([
        "golden/**/*",
        "plan.yml",
        "templates/*",
    ]),
    tags = ["manual"],
    deps = ["//tools/analysis"],
)
