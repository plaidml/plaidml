# Copyright 2019 Intel Corporation.

load("//tools/py_cffi:build_defs.bzl", "py_cffi")
load("//tools/py_setup:build_defs.bzl", "py_setup")
load("@rules_python//python:defs.bzl", "py_binary")
load("@io_bazel_rules_jsonnet//jsonnet:jsonnet.bzl", "jsonnet_to_json")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_shlib")

plaidml_cc_library(
    name = "plaidml",
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml/core",
        "//plaidml/edsl",
        "//plaidml/exec",
        "//plaidml/op",
    ],
)

jsonnet_to_json(
    name = "plaidml_link",
    src = "plaidml_link.jsonnet",
    outs = [
        "plaidml.def",
        "plaidml.ld",
        "plaidml.lds",
    ],
    extra_args = ["-S"],
)

plaidml_cc_shlib(
    name = "shlib",
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [
            "-Wl,-exported_symbols_list",
            "$(location :plaidml.ld)",
        ],
        "//conditions:default": [
            # Prefer internal library definitions.
            "-Wl,-Bsymbolic-functions",
            # Use an explicit linker script.
            "-Wl,--version-script",
            "$(location :plaidml.lds)",
        ],
    }),
    shlib_name = "plaidml",
    visibility = ["//visibility:public"],
    win_def_file = ":plaidml.def",
    deps = [
        ":plaidml",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [
            ":plaidml.ld",
        ],
        "//conditions:default": [
            ":plaidml.lds",
        ],
    }),
)

py_library(
    name = "py",
    srcs = [
        "__init__.py",
        "ffi.py",
    ],
    data = [
        ":shlib",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":_ffi",
        "//plaidml/core:py",
    ],
)

py_cffi(
    name = "_ffi",
    module = "plaidml._ffi",
    srcs_ordered = [
        "//plaidml/core:ffi.h",
        "//plaidml/edsl:ffi.h",
        "//plaidml/exec:ffi.h",
        "//plaidml/op:ffi.h",
    ],
)

py_binary(
    name = "setup",
    srcs = ["plaidml_setup.py"],
    main = "plaidml_setup.py",
    deps = [
        ":py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
        "//plaidml/op:py",
    ],
)

py_binary(
    name = "py_setup",
    srcs = ["setup.py"],
    data = [
        "plaidml-config.cmake",
        ":setup",
        "//:LICENSE",
        "//plaidml/core:sdk",
        "//plaidml/edsl:sdk",
        "//plaidml/exec:sdk",
        "//plaidml/op:sdk",
        "@easylogging//:LICENSE",
        "@gmock//:googlemock/LICENSE",
        "@llvm-project//llvm:LICENSE.TXT",
        "@llvm-project//mlir:LICENSE.TXT",
        "@xsmm//:LICENSE.md",
    ],
    main = "setup.py",
    deps = [
        ":py",
        "//plaidml/core:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
        "//plaidml/op:py",
        "//tools/py_setup",
    ],
)

py_setup(
    name = "wheel",
    package_name = "plaidml",
    platform = select({
        "@bazel_tools//src/conditions:darwin_x86_64": "macosx_10_10_x86_64",
        "@bazel_tools//src/conditions:windows": "win_amd64",
        "//conditions:default": "manylinux1_x86_64",
    }),
    tool = ":py_setup",
    universal = True,
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "testenv",
    testonly = 1,
    srcs = ["testenv.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":plaidml",
        "@gmock//:gtest",
    ],
    alwayslink = 1,
)

py_binary(
    name = "interact",
    srcs = ["interact.py"],
    deps = [
        ":py",
        "//plaidml/core:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
        "//plaidml/op:py",
    ],
)
