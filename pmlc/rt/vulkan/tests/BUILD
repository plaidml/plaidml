# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//pmlc:lit.bzl", "glob_lit_tests", "lit_test")
load("//bzl:plaidml.bzl", "plaidml_cc_test")

glob_lit_tests(
    data = ["//pmlc/rt/vulkan:vkprobe"],
)

plaidml_cc_test(
    name = "cc_test_gpu",
    srcs = ["edsl_test_gpu.cc"],
    tags = ["manual"],  # the `edsl_test.cc.test` rule will run this test via lit
    deps = [
        "//plaidml:testenv",
        "//plaidml/edsl",
        "//plaidml/exec",
        "//pmlc/rt/vulkan:link",
    ],
)

lit_test(
    name = "edsl_test_gpu.cc",
    data = [":cc_test_gpu"],
)