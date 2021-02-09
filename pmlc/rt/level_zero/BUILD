# Copyright 2021 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "level_zero",
    srcs = [
        "level_zero_device.cc",
        "level_zero_device.h",
        "level_zero_invocation.cc",
        "level_zero_invocation.h",
        "level_zero_runtime.cc",
        "level_zero_utils.cc",
        "level_zero_utils.h",
        "level_zero_wrappers.cc",
    ],
    hdrs = ["register.h"],
    linkstatic = 1,
    deps = [
        "//pmlc/compiler",
        "//pmlc/rt",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVConversion",
        "@llvm-project//mlir:mlir_runner_utils",
        "@Level_Zero//:ze_loader",
    ],
    alwayslink = 1,
)

plaidml_cc_binary(
    name = "level_zero_probe",
    srcs = [
        "level_zero_probe.cc",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": ["level_zero_utils.cc", "level_zero_utils.h",],
    }),
    defines = select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": ["PML_ENABLE_LEVEL_ZERO"],
    }),
    deps = [
        "//pmlc/util",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": ["@Level_Zero//:ze_loader"],
    }),
)

cc_test(
    name = "level_zero_probe_test",
    size = "small",
    srcs = [
        "level_zero_probe.cc",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": ["level_zero_utils.cc", "level_zero_utils.h",],
    }),
    defines = select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": ["PML_ENABLE_LEVEL_ZERO"],
    }),
    deps = [
        "//pmlc/util",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": ["@Level_Zero//:ze_loader"],
    }),
    tags = [
        "manual",
        "skip_windows",
        "skip_macos",
    ],
)
