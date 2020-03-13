# Copyright 2020 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:tblgen.bzl", "COPTS")

plaidml_cc_library(
    name = "LegalizeGpuOpForGpuLowering",
    srcs = [
        "LegalizeGpuOpForGpuLowering.cc",
        "LegalizeGpuOpForGpuLowering.h",
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        "//pmlc/util",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:GPUToSPIRVTransforms",
    ],
    alwayslink = 1,
)
