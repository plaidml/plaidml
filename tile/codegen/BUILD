load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_proto_library")

plaidml_proto_library(
    name = "proto",
    srcs = ["codegen.proto"],
    visibility = ["//visibility:public"],
    deps = ["//tile/stripe:proto"],
)

plaidml_cc_library(
    name = "codegen_base",
    srcs = glob([
        "*.cc",
        "*.h",
    ], exclude=["jit.h", "jit.cc"]),
    copts = [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":proto_cc",
        "//base/util",
        "//tile/stripe",
        "//tile/bilp",
        "@gmock//:gtest",
    ],
)

plaidml_cc_library(
    name = "codegen",
    srcs = [
        "jit.cc",
        "jit.h",
    ],
    copts = [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
    ],
    tags = ["llvm"],
    visibility = ["//visibility:public"],
    deps = [
        ":codegen_base",
        "//vendor/llvm",
        "@half",
    ],
)

