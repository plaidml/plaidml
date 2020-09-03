# Copyright 2019 Intel Corporation.

load("//tools/py_cffi:build_defs.bzl", "py_cffi")
load("//tools/py_setup:build_defs.bzl", "py_setup")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@io_bazel_rules_jsonnet//jsonnet:jsonnet.bzl", "jsonnet_to_json")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_shlib", "plaidml_settings", "plaidml_target_test_package")

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
    name = "plaidml",
    visibility = ["//visibility:public"],
    deps = [
        ":api",
        ":targets",
    ],
)

plaidml_cc_library(
    name = "targets",
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc:all_passes_and_dialects",
        "//pmlc/target/demo",
        "//pmlc/target/intel_gen",
        "//pmlc/target/x86",
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

plaidml_cc_shlib(
    name = "shlib",
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
    shlib_name = "plaidml",
    visibility = ["//visibility:public"],
    win_def_file = ":plaidml.def",
    deps = [
        ":plaidml",
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

py_library(
    name = "py",
    srcs = [
        "__init__.py",
        "ffi.py",
    ],
    data = [
        ":shlib",
    ],
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
    testonly = 1,
    srcs = ["testenv.cc"],
    hdrs = ["testenv.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":plaidml",
        "@gflags",
        "@gmock//:gtest",
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

selects.config_setting_group(
    name = "gpu_target",
    match_any = [":intel_gen"],
)

plaidml_settings(
    name = "settings",
    visibility = ["//visibility:public"],
)

plaidml_target_test_package(
    name = "intel_gen_test",
    plaidml_target = "intel_gen",
    tags = ["skip_windows"],
    tests = [
        "//plaidml/edsl/tests:cc_test",
    ],
)
