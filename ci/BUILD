load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

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
)

py_binary(
    name = "timeline",
    srcs = ["timeline.py"],
    tags = ["manual"],
)
