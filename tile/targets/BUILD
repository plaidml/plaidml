package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//tools/gencfg:build_defs.bzl", "gencfg")

gencfg(
    name = "configs",
    srcs = [
        "//tile/targets/gpu:amd.json",
        "//tile/targets/gpu:nvidia.json",
        "//tile/targets/gpu:intel_gen9.json",
    ] + select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": [
            "//tile/targets/cpu:cpu.json",
        ],
    }),
    identifier = "vertexai::tile::targets::kConfigs",
)

plaidml_cc_library(
    name = "targets",
    srcs = [
        "targets.cc",
        ":configs",
    ],
    hdrs = [
        "targets.h",
    ],
    deps = [
        "//tile/codegen",
    ] + select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": [
            "//tile/targets/cpu",
        ],
    }),
    alwayslink = 1,
)
