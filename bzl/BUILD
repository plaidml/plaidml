package(default_visibility = ["//visibility:public"])

exports_files([
    "setup.tpl.py",
    "venv.tpl.py",
    "version.tpl.cc",
])

config_setting(
    name = "android",
    values = {
        "android_crosstool_top": "@androidndk//:toolchain-libcpp",
    },
)

config_setting(
    name = "with_callgrind",
    define_values = {"with_callgrind": "true"},
)
