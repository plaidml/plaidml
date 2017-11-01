# Copyright Vertex.AI.

package(default_visibility = ["//visibility:public"])

load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_binary",
    "plaidml_cc_library",
    "plaidml_cc_test",
    "plaidml_objc_library",
    "plaidml_objc_cc_library",
    "plaidml_proto_library",
    "plaidml_py_binary",
    "plaidml_py_library",
    "plaidml_py_test",
    "plaidml_py_wheel",
    "plaidml_cc_version",
)
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar", "pkg_deb")

plaidml_proto_library(
    name = "proto",
    srcs = ["plaidml.proto"],
    visibility = ["//visibility:public"],
)

plaidml_cc_version(
    name = "version",
    prefix = "PLAIDML",
)

plaidml_cc_library(
    name = "api",
    srcs = [
        "plaidml.cc",
        ":version",
    ],
    hdrs = [
        "plaidml.h",
        "plaidml++.h",
    ],
    deps = [
        ":proto_cc",
        "//base/config",
        "//base/eventing/file",
        "//external:half",
        "//external:minizip",
        "//plaidml/base",
        "//tile/base",
        "//tile/base:program_cache",
        "//tile/hal/opencl",
        "//tile/platform/local_machine",
        "//tile/proto:proto_cc",
    ],
)

# The PLAIDML C library, defined as a Bazel dynamically-loaded library.
# This allows the library to be used as a data dependency in Bazel rules.
plaidml_cc_binary(
    name = "libplaidml.so",
    srcs = select({
        "//bzl:x64_windows": ["dummy.cc"],  # Work around for https://github.com/bazelbuild/bazel/issues/4003
        "//conditions:default": [],
    }),
    linkshared = 1,
    deps = [
        ":api",
    ],
)

plaidml_cc_binary(
    name = "plaidml.dll",
    srcs = select({
        "//bzl:x64_windows": ["dummy.cc"],  # Work around for https://github.com/bazelbuild/bazel/issues/4003
        "//conditions:default": [],
    }),
    linkshared = 1,
    deps = [
        ":api",
    ],
)

plaidml_objc_library(
    name = "api_objc",
    srcs = ["plaidml.cc"],
    tags = ["manual"],
    deps = [
        ":plaidml_cc_deps_wrap",
        "//plaidml/dev/metal:metal_objc",
    ],
)

plaidml_objc_cc_library(
    name = "api_cc_deps_wrap",
    cc_library = ":plaidml_cc_deps",
    tags = ["manual"],
    visibility = ["//visibility:private"],
)

plaidml_cc_library(
    name = "api_cc_deps",
    visibility = ["//visibility:private"],
    deps = [
        ":proto_cc",
        "//tile/base",
        "//tile/base:program_cache",
        "//tile/proto:proto_cc",
    ],
)

plaidml_cc_test(
    name = "plaidml_test",
    srcs = ["plaidml_test.cc"],
    deps = [
        ":api",
        "//testing:matchers",
        "//testing:plaidml_config",
    ],
)

plaidml_cc_test(
    name = "dtype_vec_test",
    srcs = ["dtype_vec_test.cc"],
    deps = [
        ":api",
        "//external:half",
        "//plaidml:proto_cc",
        "//testing:matchers",
        "//testing:plaidml_config",
        "//tile/hal/opencl:proto_cc",
        "//tile/platform/local_machine:proto_cc",
    ],
)

plaidml_cc_test(
    name = "matmul_fuzz_test",
    timeout = "eternal",
    srcs = ["matmul_fuzz_test.cc"],
    deps = [
        ":api",
        "//testing:plaidml_config",
    ],
)

plaidml_cc_test(
    name = "uint8_test",
    srcs = ["uint8_test.cc"],
    deps = [
        ":api",
        "//testing:matchers",
        "//testing:plaidml_config",
    ],
)

plaidml_cc_test(
    name = "cpp_test",
    srcs = ["cpp_test.cc"],
    deps = [
        ":api",
        "//testing:plaidml_config",
    ],
)

plaidml_cc_test(
    name = "network_test",
    size = "large",
    srcs = ["network_test.cc"],
    copts = select({
        "//bzl:with_callgrind": ["-DWITH_CALLGRIND"],
        "//conditions:default": [],
    }),
    data = [
        "testdata/resnet50.tpb",
        "testdata/xception.tpb",
    ],
    tags = ["manual"],
    deps = [
        ":api",
        "//testing:plaidml_config",
    ],
)

plaidml_objc_library(
    name = "plaidml_lib_objc",
    tags = ["manual"],
    deps = [
        ":plaidml_objc",
    ],
)

filegroup(
    name = "configs",
    srcs = [
        "config.json",
        "experimental.json",
    ],
)

plaidml_py_binary(
    name = "setup",
    srcs = ["plaidml_setup.py"],
    main = "plaidml_setup.py",
    deps = [":py"],
)

plaidml_py_library(
    name = "py",
    srcs = [
        "__init__.py",
        "context.py",
        "exceptions.py",
        "library.py",
        "plaidml_setup.py",
        "settings.py",
    ],
    data = [":configs"] + select({
        "//bzl:x64_windows": [":plaidml.dll"],
        "//conditions:default": [":libplaidml.so"],
    }),
    visibility = ["//visibility:public"],
)

plaidml_py_test(
    name = "settings_test",
    srcs = ["settings_test.py"],
    deps = [
        ":py",
    ],
)

plaidml_py_test(
    name = "py_plaidml_test",
    srcs = ["plaidml_test.py"],
    main = "plaidml_test.py",
    deps = [
        ":py",
        "//testing:plaidml_py",
    ],
)

pkg_tar(
    name = "pkg",
    srcs = glob([
        "**/*",
    ]),
    package_dir = "plaidml",
    strip_prefix = ".",
)

pkg_tar(
    name = "py_deb_data",
    srcs = [":py"],
    package_dir = "/usr/lib/python2.7/dist-packages/plaidml",
)

pkg_deb(
    name = "python-plaidml",
    data = ":py_deb_data",
    depends = [
        "python",
    ],
    description = "Vertex.AI PlaidML Python",
    homepage = "https://www.vertex.ai",
    maintainer = "eng@vertex.ai",
    package = "python-plaidml",
    priority = "extra",
    section = "non-free/misc",
    tags = ["deb"],
    version = "0.0.1",
)

plaidml_py_wheel(
    name = "wheel",
    srcs = [
        ":py",
        ":configs",
    ] + select({
        "//bzl:x64_windows": ["//plaidml:plaidml.dll"],
        "//conditions:default": ["//plaidml:libplaidml.so"],
    }),
    config = ":setup.cfg",
    console_scripts = [
        "'plaidml-setup = plaidml.plaidml_setup:main'",
    ],
    package = "plaidml",
    platform = select({
        "//bzl:arm7-ubuntu-14.04": "linux_armv7l",
        "//bzl:darwin": "macosx_10_10_x86_64",
        "//bzl:x64_windows": "win_amd64",
        "//conditions:default": "manylinux1_x86_64",
    }),
    python = "py2.py3",
)
