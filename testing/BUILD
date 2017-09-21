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
    data = select({
        "//conditions:default": glob(["*.json"]),
        "//bzl:x64_windows": glob(
            ["*.json"],
            exclude = ["plaidml_llvm.json"],
        ),
    }),
)

pkg_tar(
    name = "pkg",
    files = glob(["*.json"]),
    package_dir = "testing",
    strip_prefix = ".",
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

test_suite(
    name = "plaidml_tests",
    tests = [
        "//base/context:context_test",
        "//base/context:gate_test",
        "//base/util/astgen:ast_test",
        "//plaidml:cpp_test",
        "//plaidml:plaidml_test",
        "//plaidml:py_plaidml_test",
        "//plaidml:uint8_test",
        "//plaidml/base:feature_test",
        "//plaidml/base:status_test",
        "//plaidml/keras:backend_test",
        "//plaidml/keras:mnist_mlp_test",
        "//plaidml/keras:trivial_model_test",
        "//tile/lang:gid_test",
        "//tile/lang:test",
        "//tile/platform/local_machine:platform_test",
    ],
)
