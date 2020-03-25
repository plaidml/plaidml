package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_test")

licenses(["notice"])  # BSD/MIT-like license

plaidml_cc_test(
    name = "smoke",
    tags = [
        "skip_windows",
    ],
    deps = [
        "@openvino//:smoke_tests",
    ],
)
