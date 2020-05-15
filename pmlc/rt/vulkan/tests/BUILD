# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//pmlc:lit.bzl", "glob_lit_tests", "lit_test")
load("//bzl:plaidml.bzl", "plaidml_cc_test","plaidml_cc_library")

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
    tags = ["manual"],  # the `edsl_test.cc.test` rule will run this test via lit
    deps = [
        ":testenv",
        "//plaidml/edsl",
        "//plaidml/exec",
        "//pmlc/rt/vulkan",
    ],
    linkstatic = 1,
)

lit_test(
	name="edsl_test_gpu.cc",
    data = [
    	":cc_test_gpu",
    	"lit.local.cfg",
    	"//pmlc/rt/vulkan:vkprobe",
    ],
)