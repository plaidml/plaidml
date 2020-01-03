# Copyright 2019 Intel Corporation.

load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_binary",
    "plaidml_cc_test",
)
load("//vendor/mlir:tblgen.bzl", "COPTS")

plaidml_cc_binary(
    name = "pmlc-opt",
    srcs = ["pmlc-opt.cc"],
    copts = COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/conversion/affine_to_stripe",
        "//pmlc/conversion/stripe_to_affine",
        "//pmlc/dialect/eltwise",
        "//pmlc/dialect/pxa",
        "//pmlc/dialect/stripe",
        "//pmlc/dialect/stripe:passes",
        "//pmlc/dialect/tile",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Transforms",
    ],
)

plaidml_cc_binary(
    name = "pmlc-translate",
    srcs = ["pmlc-translate.cc"],
    copts = COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/dialect/stripe:transcode",
        "//pmlc/dialect/tile",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateClParser",
        "@llvm-project//mlir:Translation",
    ],
)

filegroup(
    name = "litfiles",
    srcs = glob(["runlit*py"]),
    visibility = ["//visibility:public"],
)

plaidml_cc_test(
    name = "test",
    srcs = ["test.cc"],
    copts = COPTS,
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
    ],
)
