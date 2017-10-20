load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_py_library", "plaidml_py_test", "run_as_test")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "matchers",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        "//base/util:util_cc",
        "//external:gtest",
        "//plaidml/base",
    ],
)

plaidml_cc_library(
    name = "runfiles_db",
    testonly = True,
    srcs = ["runfiles_db.cc"],
    hdrs = ["runfiles_db.h"],
)

plaidml_cc_library(
    name = "plaidml_config",
    testonly = True,
    srcs = ["plaidml_config.cc"],
    hdrs = ["plaidml_config.h"],
    data = [":configs"],
    deps = [
        ":runfiles_db",
        "//base/util:util_cc",
    ],
)

plaidml_py_library(
    name = "plaidml_py",
    testonly = True,
    srcs = [
        "__init__.py",
        "plaidml_config.py",
    ],
    data = [":configs"],
)

filegroup(
    name = "configs",
    data = glob(
        ["*.json"],
        exclude = ["tile_llvm.json"],
    ),
)

plaidml_py_test(
    name = "configs_test",
    srcs = ["configs_test.py"],
    data = [":configs"],
    deps = ["//plaidml:py"],
)

plaidml_cc_library(
    name = "gtest_main",
    testonly = True,
    srcs = ["gtest_main.cc"],
    deps = [
        "//base/util:util_cc",
        "//external:easylogging",
        "//external:gflags",
        "//external:gtest",
    ],
)
