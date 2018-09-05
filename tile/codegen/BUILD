load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "codegen",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//tile/lang",
        "//tile/proto:stripe_cc",
    ],
)
