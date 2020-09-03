# Copyright 2020 Intel Corporation

load("@rules_python//python:defs.bzl", "py_test")
load("//bzl:plaidml.bzl", "plaidml_cc_test")
load("//pmlc:lit.bzl", "lit_test")

plaidml_cc_test(
    name = "cc_test",
    srcs = ["edsl_test.cc"],
    args = [
        "--plaidml_device=$(plaidml_device)",
        "--plaidml_target=$(plaidml_target)",
    ] + select({
        "//plaidml:intel_gen": ["--skip_test_file=$(location edsl_test-skip.intel_gen.txt)"],
        "//conditions:default": [],
    }),
    data = select({
        "//plaidml:intel_gen": [":edsl_test-skip.intel_gen.txt"],
        "//conditions:default": [],
    }),
    linkstatic = True,
    toolchains = ["//plaidml:settings"],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml:testenv",
        "//plaidml/edsl",
        "//plaidml/exec",
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
