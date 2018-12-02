load("//bzl:conda.bzl", "conda_binary")

package(default_visibility = ["//visibility:public"])

exports_files([
    "conda.tpl.py",
    "conda_wrapper.py",
    "setup.tpl.py",
    "sphinx.py",
    "version.tpl.cc",
])

config_setting(
    name = "with_callgrind",
    define_values = {"with_callgrind": "true"},
)

conda_binary(
    name = "sphinx",
    srcs = [":sphinx.py"],
    env = "@com_intel_plaidml_conda_sphinx//:env",
)
