# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_shlib")

plaidml_cc_library(
    name = "rt",
    srcs = glob(["*.cc"]),
    visibility = ["//visibility:public"],
    deps = ["@llvm-project//llvm:support"],
    alwayslink = 1,
)

plaidml_cc_shlib(
    name = "shlib",
    shlib_name = "plaidml_rt",
    visibility = ["//visibility:public"],
    deps = [":rt"],
)

plaidml_cc_library(
    name = "rt_shlib",
    srcs = [":shlib"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
