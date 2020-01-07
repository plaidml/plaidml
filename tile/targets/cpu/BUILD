# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_jsonnet//jsonnet:jsonnet.bzl", "jsonnet_to_json")
load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//tools/heatmap:build_defs.bzl", "heatmap")

jsonnet_to_json(
    name = "llvm_cpu_jsonnet",
    src = "llvm_cpu.jsonnet",
    outs = ["llvm_cpu.json"],
)

plaidml_cc_library(
    name = "cpu",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = ["heatmap.tpl.cc"],
    ) + [
        ":heatmap",
    ],
    tags = ["llvm"],
    deps = [
        "//tile/stripe",
        "@half",
        "@llvm-project//llvm:execution_engine",
        "@llvm-project//llvm:ipo",
        "@llvm-project//llvm:mcjit",
        "@llvm-project//llvm:x86_asm_parser",
        "@llvm-project//llvm:x86_code_gen",
        "@tbb",
        "@xsmm",
    ],
)

heatmap(
    name = "heatmap",
    out = "heatmap.cc",
    csv = "heatmap_skx_xeonplat_8180_1-7GHz_mblocked.csv.gz",
    template = "heatmap.tpl.cc",
)
