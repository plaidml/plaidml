# Copyright 2018 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:conda.bzl", "conda_binary", "conda_test")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_binary",
    "plaidml_cc_library",
    "plaidml_cc_test",
    "plaidml_cc_version",
    "plaidml_proto_library",
    "plaidml_py_library",
    "plaidml_py_wheel",
)
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_deb", "pkg_tar")
load("@cuda//:build_defs.bzl", "if_cuda_is_configured")

exports_files(["requirements.txt"])

plaidml_proto_library(
    name = "proto",
    srcs = ["plaidml.proto"],
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
        "//base/util:runfiles_db",
        "//plaidml/base",
        "//tile/base",
        "//tile/base:program_cache",
        "//tile/stripe",
        "//tile/platform/local_machine",
        "//tile/proto:metadata_cc",
        "@boost//:filesystem",
        "@half",
        "@minizip",
    ] + select({
        "@toolchain//:linux_arm_32v7": [
            "//tile/hal/opencl",
        ],
        "@toolchain//:linux_arm_64v8": [
            "//tile/hal/opencl",
        ],
        "@toolchain//:macos_x86_64": [
            "//tile/hal/cpu",
            "//tile/hal/metal",
            "//tile/hal/opencl",
        ],
        "@toolchain//:windows_x86_64": [
            "//tile/hal/opencl",
        ],
        "//conditions:default": [
            "//tile/hal/cpu",
            "//tile/hal/opencl",
        ],
    }) + if_cuda_is_configured([
        "//tile/hal/cuda",
    ]),
    alwayslink = 1,
)

# The PLAIDML C library, defined as a Bazel dynamically-loaded library.
# This allows the library to be used as a data dependency in Bazel rules.
plaidml_cc_binary(
    name = "libplaidml.so",
    srcs = select({
        "@toolchain//:windows_x86_64": ["dummy.cc"],  # Work around for https://github.com/bazelbuild/bazel/issues/4003
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
        "@toolchain//:windows_x86_64": ["dummy.cc"],  # Work around for https://github.com/bazelbuild/bazel/issues/4003
        "//conditions:default": [],
    }),
    linkshared = 1,
    deps = [
        ":api",
    ],
)

apple_binary(
    name = "libplaidml.dylib",
    platform_type = "macos",
    binary_type = "dylib",
    deps = [":api"],
    tags = ["darwin"],
)

plaidml_cc_library(
    name = "api_cc_deps",
    visibility = ["//visibility:private"],
    deps = [
        ":proto_cc",
        "//tile/base",
        "//tile/base:program_cache",
        "//tile/stripe",
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

genrule(
    name = "configs",
    srcs = select({
        "@toolchain//:macos_x86_64": [
            "configs/empty.json",
            "configs/macos.json",
        ],
        "//conditions:default": [
            "configs/config.json",
            "configs/experimental.json",
        ],
    }),
    outs = [
        "config.json",
        "experimental.json",
    ],
    cmd = """
    srcs=($(SRCS))
    outs=($(OUTS))
    cp $${srcs[0]} $${outs[0]}
    cp $${srcs[1]} $${outs[1]}
""",
)

conda_binary(
    name = "enumerate",
    srcs = ["enumerate.py"],
    env = "@com_intel_plaidml_conda//:env",
    deps = [":py"],
)

conda_binary(
    name = "setup",
    srcs = ["plaidml_setup.py"],
    env = "@com_intel_plaidml_conda//:env",
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
        "op.py",
        "plaidml_setup.py",
        "settings.py",
        "tile.py",
    ],
    data = [":configs"] + select({
        "@toolchain//:windows_x86_64": [":plaidml.dll"],
        "//conditions:default": [":libplaidml.so"],
    }),
)

conda_test(
    name = "settings_test",
    srcs = ["settings_test.py"],
    env = "@com_intel_plaidml_conda//:env",
    deps = [
        ":py",
    ],
)

conda_test(
    name = "py_plaidml_test",
    srcs = ["plaidml_test.py"],
    env = "@com_intel_plaidml_conda//:env",
    main = "plaidml_test.py",
    deps = [
        ":py",
        "//testing:plaidml_py",
    ],
)

conda_test(
    name = "py_tile_test",
    srcs = ["tile_test.py"],
    env = "@com_intel_plaidml_conda//:env",
    main = "tile_test.py",
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
    package_name = "plaidml",
    srcs = [
        ":py",
        ":configs",
    ] + select({
        "@toolchain//:windows_x86_64": ["//plaidml:plaidml.dll"],
        "//conditions:default": ["//plaidml:libplaidml.so"],
    }),
    config = ":setup.cfg",
    package_prefix = "plaidml",
    platform = select({
        "@toolchain//:linux_arm_32v7": "linux_armv7l",
        "@toolchain//:linux_arm_64v8": "linux_aarch64",
        "@toolchain//:macos_x86_64": "macosx_10_10_x86_64",
        "@toolchain//:windows_x86_64": "win_amd64",
        "//conditions:default": "manylinux1_x86_64",
    }),
    python = "py2.py3",
)

pkg_tar(
    name = "sdk_configs",
    srcs = [":configs"],
    package_dir = "share/plaidml",
)

pkg_tar(
    name = "sdk_includes",
    srcs = [
        "plaidml.h",
        "plaidml++.h",
    ],
    package_dir = "include/plaidml",
)

pkg_tar(
    name = "macos_sdk_bin",
    srcs = [
        "update_sdk_rpaths"
    ],
    package_dir = "bin",
    tags = ["darwin"],
)

pkg_tar(
    name = "sdk_libs",
    srcs = select({
        "@toolchain//:windows_x86_64": [":plaidml.dll"],
        "@toolchain//:macos_x86_64": [":libplaidml.dylib"],
        "//conditions:default": ["libplaidml.so"],
    }),
    package_dir = "lib",
    symlinks = select({
        "@toolchain//:macos_x86_64": {
            "lib/libplaidml.dylib": "libplaidml.dylib_lipobin",
        },
        "//conditions:default": {},
    }),
)

pkg_tar(
    name = "sdk",
    srcs = ["plaidml-config.cmake"],
    # TODO: Build package_dir from the --define=version= config setting.
    package_dir = "plaidml-0.5.0",
    deps = [
        ":sdk_configs",
        ":sdk_includes",
        ":sdk_libs",
        "//plaidml/base:sdk_includes",
        "@half//:sdk_includes",
    ] + select({
        "@toolchain//:macos_x86_64": [":macos_sdk_bin"],
        "//conditions:default": [],
    }),
)
