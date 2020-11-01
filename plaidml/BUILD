# Copyright 2019 Intel Corporation.

load("//tools/py_cffi:build_defs.bzl", "py_cffi")
load("//tools/py_setup:build_defs.bzl", "py_setup")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@io_bazel_rules_jsonnet//jsonnet:jsonnet.bzl", "jsonnet_to_json")
load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library", "plaidml_settings")
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

plaidml_cc_library(
    name = "api",
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml/core",
        "//plaidml/edsl",
        "//plaidml/exec",
        "//plaidml/op",
    ],
)

plaidml_cc_library(
    name = "lib",
    visibility = ["//visibility:public"],
    deps = [
        ":api",
        ":runtimes",
        ":targets",
    ],
)

plaidml_cc_library(
    name = "targets",
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/target/intel_gen",
        "//pmlc/target/intel_gen_ocl_spirv",
        "//pmlc/target/x86",
    ],
)

plaidml_cc_library(
    name = "runtimes",
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/rt/llvm",
        "//pmlc/rt/vulkan",
    ],
)

jsonnet_to_json(
    name = "plaidml_link",
    src = "plaidml_link.jsonnet",
    outs = [
        "plaidml.def",
        "plaidml.ld",
        "plaidml.lds",
    ],
    extra_args = ["-S"],
)

plaidml_cc_binary(
    name = "plaidml",
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [
            "-Wl,-exported_symbols_list",
            "$(location :plaidml.ld)",
        ],
        "//conditions:default": [
            # Prefer internal library definitions.
            "-Wl,-Bsymbolic-functions",
            # Use an explicit linker script.
            "-Wl,--version-script",
            "$(location :plaidml.lds)",
        ],
    }),
    linkshared = 1,
    visibility = ["//visibility:public"],
    win_def_file = ":plaidml.def",
    deps = [
        ":lib",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [
            ":plaidml.ld",
        ],
        "//conditions:default": [
            ":plaidml.lds",
        ],
    }),
)

pkg_tar(
    name = "sdk_tar",
    srcs = [
        ":plaidml",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml/core:sdk_tar",
        "//plaidml/edsl:sdk_tar",
        "//plaidml/exec:sdk_tar",
        "//plaidml/op:sdk_tar",
    ],
)

py_library(
    name = "py",
    srcs = [
        "__init__.py",
        "ffi.py",
    ],
    data = [":plaidml"],
    visibility = ["//visibility:public"],
    deps = [
        ":_ffi",
        "//plaidml/core:py",
    ],
)

py_cffi(
    name = "_ffi",
    module = "plaidml._ffi",
    srcs_ordered = [
        "//plaidml/core:ffi.h",
        "//plaidml/edsl:ffi.h",
        "//plaidml/exec:ffi.h",
        "//plaidml/op:ffi.h",
    ],
)

py_binary(
    name = "setup",
    srcs = ["plaidml_setup.py"],
    main = "plaidml_setup.py",
    deps = [
        ":py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
        "//plaidml/op:py",
    ],
)

py_binary(
    name = "py_setup",
    srcs = ["setup.py"],
    data = [
        "plaidml-config.cmake",
        ":setup",
        "//:LICENSE",
        "//bzl:googlemock.LICENSE",
        "//plaidml/core:sdk",
        "//plaidml/edsl:sdk",
        "//plaidml/exec:sdk",
        "//plaidml/op:sdk",
        "@boost//:LICENSE_1_0.txt",
        "@easylogging//:LICENSE",
        "@half//:LICENSE.txt",
        "@llvm-project//llvm:LICENSE.TXT",
        "@llvm-project//mlir:LICENSE.TXT",
        "@xsmm//:LICENSE.md",
    ],
    main = "setup.py",
    deps = [
        ":py",
        "//plaidml/core:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
        "//plaidml/op:py",
        "//tools/py_setup",
    ],
)

py_setup(
    name = "wheel",
    package_name = "plaidml",
    platform = select({
        "@bazel_tools//src/conditions:darwin_x86_64": "macosx_10_10_x86_64",
        "@bazel_tools//src/conditions:windows": "win_amd64",
        "//conditions:default": "manylinux1_x86_64",
    }),
    tool = ":py_setup",
    universal = True,
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "testenv",
    srcs = ["testenv.cc"],
    hdrs = ["testenv.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
        "@com_google_googletest//:gtest",
        "@gflags",
    ],
    alwayslink = 1,
)

py_binary(
    name = "interact",
    srcs = ["interact.py"],
    deps = [
        ":py",
        "//plaidml/core:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
        "//plaidml/op:py",
    ],
)

string_flag(
    name = "device",
    build_setting_default = "llvm_cpu.0",
)

string_flag(
    name = "target",
    build_setting_default = "llvm_cpu",
)

config_setting(
    name = "llvm_cpu",
    flag_values = {
        ":target": "llvm_cpu",
    },
)

config_setting(
    name = "intel_gen",
    flag_values = {
        ":target": "intel_gen",
    },
)

config_setting(
    name = "intel_gen_ocl_spirv",
    flag_values = {
        ":target": "intel_gen_ocl_spirv",
    },
)

selects.config_setting_group(
    name = "gpu_target",
    match_any = [
        ":intel_gen",
        ":intel_gen_ocl_spirv",
    ],
)

plaidml_settings(
    name = "settings",
    visibility = ["//visibility:public"],
)
