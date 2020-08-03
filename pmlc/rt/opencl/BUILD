# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

plaidml_cc_library(
    name = "opencl",
    srcs = [
        "opencl_runtime.cc",
        "wrappers.cc",
    ],
    hdrs = [
        "opencl_runtime.h",
    ],
    deps = [
        "//pmlc/compiler",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVLowering",
        "@llvm-project//mlir:mlir_runner_utils",
        "@opencl_hpp_headers",
        "@opencl_icd_loader"
    ],
    alwayslink = 1,
)


plaidml_cc_binary(
    name = "clprobe",
    srcs = ["clprobe.cc"],
    deps = ["@opencl_icd_loader", "@opencl_hpp_headers"],
)
