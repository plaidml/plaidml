# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_jsonnet//jsonnet:jsonnet.bzl", "jsonnet_to_json")
load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library", "plaidml_cc_shlib")

plaidml_cc_library(
    name = "vulkan",
    srcs = [
        "vulkan_runtime.cc",
        "wrappers.cc",
    ],
    hdrs = [
        "vulkan_runtime.h",
    ],
    local_defines = ["VULKAN_RT_BUILD"],
    deps = [
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVLowering",
        "@volk",
    ],
    alwayslink = 1,
)

jsonnet_to_json(
    name = "link_script",
    src = "vulkan_rt.jsonnet",
    outs = [
        "vulkan_rt.def",
        "vulkan_rt.ld",
        "vulkan_rt.lds",
    ],
    extra_args = ["-S"],
)

plaidml_cc_shlib(
    name = "shlib",
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [
            "-Wl,-exported_symbols_list",
            "$(location :vulkan_rt.ld)",
        ],
        "//conditions:default": [
            # Prefer internal library definitions.
            "-Wl,-Bsymbolic-functions",
            # Use an explicit linker script.
            "-Wl,--version-script",
            "$(location :vulkan_rt.lds)",
        ],
    }),
    shlib_name = "vulkan_rt",
    visibility = ["//visibility:public"],
    win_def_file = ":vulkan_rt.def",
    deps = [
        ":vulkan",
        "//pmlc/rt",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [
            ":vulkan_rt.ld",
        ],
        "//conditions:default": [
            ":vulkan_rt.lds",
        ],
    }),
)

plaidml_cc_library(
    name = "link",
    srcs = [":shlib"],
    linkstatic = 1,
    deps = select({
        "@bazel_tools//src/conditions:windows": [":vulkan"],
        "//conditions:default": [],
    }),
)

plaidml_cc_binary(
    name = "vkprobe",
    srcs = ["vkprobe.cc"],
    deps = ["@volk"],
)
