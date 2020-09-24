load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "i3d",
    srcs = ["i3d.py"],
)

filegroup(
    name = "i3d-data",
    srcs = glob(["data/**"]),
)
