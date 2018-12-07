package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "lib",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    deps = [
        "@boost//:boost",
        "//tile/lang",
        "//tile/util",
    ],
)
