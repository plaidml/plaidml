# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//tools/bin2h:build_defs.bzl", "bin2h")
load("//tools/gencfg:build_defs.bzl", "gencfg")

gencfg(
    name = "cfg",
    target = "cpu",
    tmpl = "cpu.json.j2",
    yml = "cpu.yml",
)

bin2h(
    name = "configs",
    srcs = {
        ":cfg": "kCpu",
    },
)

plaidml_cc_library(
    name = "cpu",
    srcs = glob([
        "*.cc",
        "*.h",
    ]) + [":configs"],
    copts = [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
    ],
    tags = ["llvm"],
    deps = [
        "//tile/codegen",
        "//tile/stripe",
        "@half",
        "@llvm_shim//:llvm",
    ],
    alwayslink = 1,
)
