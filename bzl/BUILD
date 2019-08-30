load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

exports_files([
    "conda.tpl.py",
    "conda_wrapper.py",
    "setup.tpl.py",
    "sphinx.py",
    "version.tpl.cc",
    "version.tpl.py",
])

config_setting(
    name = "with_callgrind",
    define_values = {"with_callgrind": "true"},
)

py_binary(
    name = "sphinx",
    srcs = [":sphinx.py"],
)
