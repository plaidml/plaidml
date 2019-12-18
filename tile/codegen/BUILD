load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_proto_library")
load("//vendor/mlir:tblgen.bzl", "COPTS")

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
        "//pmlc/dialect/stripe:passes",
        "//pmlc/dialect/stripe:transcode",
        "//tile/bilp",
        "//tile/stripe",
        "//tile/targets/cpu",
        "@boost//:filesystem",
    ],
    alwayslink = 1,
)
