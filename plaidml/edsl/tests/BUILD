# Copyright 2020 Intel Corporation

load("@rules_python//python:defs.bzl", "py_test")
load("//bzl:plaidml.bzl", "plaidml_cc_test")
load("//pmlc:lit.bzl", "lit_test")

exports_files([
    "edsl_test-skip.intel_gen.txt",
    "edsl_test-skip.intel_gen_ocl_spirv.txt",
])

plaidml_cc_test(
    name = "cc_test",
    size = "medium",
    testonly = 0,
    srcs = ["edsl_test.cc"],
    args = [
        "--plaidml_device=$(plaidml_device)",
        "--plaidml_target=$(plaidml_target)",
    ] + select({
        "//plaidml:intel_gen": ["--skip_test_file=$(location edsl_test-skip.intel_gen.txt)"],
        "//plaidml:intel_gen_ocl_spirv": ["--skip_test_file=$(location edsl_test-skip.intel_gen_ocl_spirv.txt)"],
        "//conditions:default": [],
    }),
    copts = select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": ["-Wno-comment"],
    }),
    data = select({
        "//plaidml:intel_gen": [":edsl_test-skip.intel_gen.txt"],
        "//plaidml:intel_gen_ocl_spirv": [":edsl_test-skip.intel_gen_ocl_spirv.txt"],
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
    size = "medium",
    srcs = ["edsl_test.py"],
    args = ["-v"],
    main = "edsl_test.py",
    deps = [
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
)

lit_test(
    name = "edsl_test.cc",
    size = "medium",
    data = [":cc_test"],
)
