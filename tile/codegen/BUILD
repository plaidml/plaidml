load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_proto_library")
load("//vendor/mlir:mlir.bzl", "COPTS")

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
    copts = COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":proto_cc",
        "//base/config",
        "//base/util",
        "//tile/bilp",
        "//tile/stripe",
        "//pmlc/dialect/stripe:transcode",
        "//pmlc/dialect/stripe:passes",
        "//tile/targets/cpu"
        "@boost//:filesystem",
    ],
    alwayslink = 1,
)
