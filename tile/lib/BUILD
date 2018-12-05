package(default_visibility = ["//visibility:public"])

load("@com_intel_plaidml//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "lib",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    deps = [
        "@com_intel_plaidml//tile/lang",
        "@com_intel_plaidml//tile/util",
    ],
)
