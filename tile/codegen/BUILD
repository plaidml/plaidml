load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_proto_library")

plaidml_proto_library(
    name = "proto",
    srcs = ["codegen.proto"],
    visibility = ["//visibility:public"],
    deps = ["//tile/stripe:proto"],
)

plaidml_cc_library(
    name = "codegen",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":proto_cc",
        "//base/config",
        "//base/util",
        "//tile/bilp",
        "//tile/stripe",
        "@boost//:filesystem",
    ] + select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": ["//tile/targets/cpu"],
    }),
    alwayslink = 1,
)
