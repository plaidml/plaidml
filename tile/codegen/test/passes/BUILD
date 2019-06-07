load("//bzl:plaidml.bzl", "plaidml_cc_binary")

plaidml_cc_binary(
    name = "passes",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//tile/codegen",
        "@boost//:filesystem",
        "@boost//:program_options",
    ],
)
