# Copyright 2020 Intel Corporation

load("@rules_python//python:defs.bzl", "py_test")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test", "plaidml_cc_binary")
load("//pmlc:lit.bzl", "lit_test")

plaidml_cc_binary(
    name = "cc_test",
    testonly = 1,
    srcs = ["edsl_test.cc"],
    tags = ["manual"],  # the `edsl_test.cc.test` rule will run this test via lit
    deps = [
        "//plaidml:testenv",
        "//plaidml/edsl",
        "//plaidml/exec",
        "//pmlc/rt/vulkan:link",
        "//pmlc/testing:gtest_main",
    ],
)

py_test(
    name = "py_test",
    srcs = [
        "__init__.py",
        "edsl_test.py",
    ],
    args = ["-v"],
    main = "edsl_test.py",
    tags = ["manual"],  # the `edsl_test.py.test` rule will run this test via lit
    deps = [
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
)

lit_test(
    name = "edsl_test.cc",
    data = [":cc_test"],
)

lit_test(
    name = "edsl_test.py",
    data = [":py_test"],
    tags = ["skip_windows"],
)
