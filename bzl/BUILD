load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

exports_files([
    "conda_wrapper.py",
    "googlemock.LICENSE",
    "version.tpl.cc",
    "version.tpl.py",
])

config_setting(
    name = "with_callgrind",
    define_values = {"with_callgrind": "true"},
)
