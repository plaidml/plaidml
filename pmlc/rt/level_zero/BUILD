# Copyright 2020 Intel Corporation

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
        "level_zero_wrappers.cc",
        "level_zero_utils.h",
        "level_zero_utils.cc",
    ],
    hdrs = ["register.h"],
    deps = [
        "//pmlc/compiler",
        "//pmlc/rt",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVLowering",
        "@llvm-project//mlir:mlir_runner_utils",
        "@Level_Zero//:ze_loader",
    ],
    alwayslink = 1,
    linkstatic = 1,
)

plaidml_cc_binary(
    name = "level_zero_probe",
    srcs = [
        "level_zero_probe.cc",
        "level_zero_utils.h",
        "level_zero_utils.cc",
    ],
    deps = [
        "@Level_Zero//:ze_loader",
        "//pmlc/util",
    ],
)

#cc_test(
#    name = "level_zero_probe_test",
#    srcs = [
#        "level_zero_probe.cc",
#        "level_zero_utils.h",
#        "level_zero_utils.cc",
#    ],
#    deps = [
#        "@Level_Zero//:ze_loader",
#        "//pmlc/util",
#    ],
#    size = "small",
#)
