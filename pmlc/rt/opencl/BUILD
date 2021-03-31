# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "opencl",
    srcs = [
        "common_wrappers.cc",
        "opencl_device.cc",
        "opencl_device.h",
        "opencl_invocation.cc",
        "opencl_invocation.h",
        "opencl_runtime.cc",
        "opencl_utils.cc",
        "opencl_utils.h",
        "opencl_wrappers.cc",
    ],
    hdrs = ["register.h"],
    linkstatic = 1,
    deps = [
        "//pmlc/compiler",
        "//pmlc/rt",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVConversion",
        "@llvm-project//mlir:mlir_runner_utils",
        "@opencl_hpp_headers",
        "@opencl_icd_loader",
    ],
    alwayslink = 1,
)

plaidml_cc_binary(
    name = "clprobe",
    srcs = [
        "clprobe.cc",
        "opencl_utils.cc",
        "opencl_utils.h",
    ],
    deps = [
        "@opencl_hpp_headers",
        "@opencl_icd_loader",
    ],
)

cc_test(
    name = "clprobe_test",
    size = "small",
    srcs = [
        "clprobe.cc",
        "opencl_utils.cc",
        "opencl_utils.h",
    ],
    deps = [
        "@opencl_hpp_headers",
        "@opencl_icd_loader",
    ],
)
