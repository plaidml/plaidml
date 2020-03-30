# Copyright 2020 Intel Corporation

load("@io_bazel_rules_jsonnet//jsonnet:jsonnet.bzl", "jsonnet_to_json")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_shlib")

plaidml_cc_library(
    name = "rt",
    srcs = [
        "boundscheck.cc",
        "memref.h",
        "runtime.cc",
        "xsmm.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:mlir_runner_utils",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "@xsmm",
        ],
    }),
    alwayslink = 1,
)

jsonnet_to_json(
    name = "link_script",
    src = "plaidml_rt.jsonnet",
    outs = [
        "plaidml_rt.def",
        "plaidml_rt.ld",
        "plaidml_rt.lds",
    ],
    extra_args = ["-S"],
)

plaidml_cc_shlib(
    name = "shlib",
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [
            "-Wl,-exported_symbols_list",
            "$(location :plaidml_rt.ld)",
        ],
        "//conditions:default": [
            # Prefer internal library definitions.
            "-Wl,-Bsymbolic-functions",
            # Use an explicit linker script.
            "-Wl,--version-script",
            "$(location :plaidml_rt.lds)",
        ],
    }),
    shlib_name = "plaidml_rt",
    visibility = ["//visibility:public"],
    win_def_file = ":plaidml_rt.def",
    deps = [
        ":rt",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [
            ":plaidml_rt.ld",
        ],
        "//conditions:default": [
            ":plaidml_rt.lds",
        ],
    }),
)

plaidml_cc_library(
    name = "link",
    srcs = select({
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": [":shlib"],
    }),
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
