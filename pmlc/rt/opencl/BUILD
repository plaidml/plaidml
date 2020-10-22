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
        "opencl_wrappers.cc",
        "opencl_utils.h",
        "opencl_utils.cc",
    ],
    deps = [
        "//pmlc/compiler",
        "//pmlc/rt",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVLowering",
        "@llvm-project//mlir:mlir_runner_utils",
        "@opencl_hpp_headers",
        "@opencl_icd_loader"
    ],
    alwayslink = 1,
    linkstatic = 1,
)

plaidml_cc_binary(
    name = "clprobe",
    srcs = [
        "clprobe.cc",
        "opencl_utils.h",
        "opencl_utils.cc",
    ],
    deps = ["@opencl_icd_loader", "@opencl_hpp_headers"],
)

cc_test(
    name = "clprobe_test",
    srcs = [
        "clprobe.cc",
        "opencl_utils.h",
        "opencl_utils.cc",
    ],
    deps = ["@opencl_icd_loader", "@opencl_hpp_headers"],
    size = "small",
)
