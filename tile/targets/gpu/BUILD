# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//tools/bin2h:build_defs.bzl", "bin2h")
load("//tools/gencfg:build_defs.bzl", "gencfg")

gencfg(
    name = "amd",
    target = "amd",
    tmpl = "gpu.json.j2",
    yml = "gpu.yml",
)

gencfg(
    name = "intel_gen9",
    target = "intel_gen9",
    tmpl = "gpu.json.j2",
    yml = "gpu.yml",
)

gencfg(
    name = "nvidia",
    target = "nvidia",
    tmpl = "gpu.json.j2",
    yml = "gpu.yml",
)

bin2h(
    name = "configs",
    srcs = {
        ":amd": "kGpuAmd",
        ":intel_gen9": "kGpuIntelGen9",
        ":nvidia": "kGpuNvidia",
    },
)

plaidml_cc_library(
    name = "gpu",
    srcs = [
        "configs.cc",
        ":configs",
    ],
    deps = ["//tile/codegen"],
    alwayslink = 1,
)
