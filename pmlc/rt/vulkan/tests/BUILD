# Copyright 2020 Intel Corporation

load("//pmlc:lit.bzl", "glob_lit_tests", "lit_test")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

package(default_visibility = ["//visibility:public"])

glob_lit_tests(
    data = ["//pmlc/rt/vulkan:vkprobe"],
)

plaidml_cc_library(
    name = "testenv",
    testonly = 1,
    srcs = ["testenv.cc"],
    hdrs = ["testenv.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml",
        "@gmock//:gtest",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "cc_test_gpu",
    srcs = ["edsl_test_gpu.cc"],
    linkstatic = 1,
    tags = ["manual"],  # the `edsl_test.cc.test` rule will run this test via lit
    deps = [
        ":testenv",
        "//plaidml/edsl",
        "//plaidml/exec",
        "//pmlc/rt/vulkan",
    ],
)

lit_test(
    name = "edsl_test_gpu.cc",
    data = [
        "lit.local.cfg",
        ":cc_test_gpu",
        "//pmlc/rt/vulkan:vkprobe",
    ],
)
