load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_proto_library")

plaidml_proto_library(
    name = "proto",
    srcs = ["codegen.proto"],
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
        "//base/util",
        "//tile/stripe",
    ],
)
