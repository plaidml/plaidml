# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_jsonnet//jsonnet:jsonnet.bzl", "jsonnet_to_json")
load("//bzl:plaidml.bzl", "plaidml_cc_library")

jsonnet_to_json(
    name = "llvm_cpu_jsonnet",
    src = "llvm_cpu.jsonnet",
    outs = ["llvm_cpu.json"],
)

jsonnet_to_json(
    name = "opencl_cpu_jsonnet",
    src = "opencl_cpu.jsonnet",
    outs = ["opencl_cpu.json"],
)

plaidml_cc_library(
    name = "cpu",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    tags = ["llvm"],
    deps = [
        "//tile/stripe",
        "@half",
        "@llvm//:execution_engine",
        "@llvm//:ipo",
        "@llvm//:mcjit",
        "@llvm//:x86_asm_parser",
        "@llvm//:x86_code_gen",
        "@tbb",
        "@xsmm",
    ],
)
