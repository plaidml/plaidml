load("@vertexai_plaidml//bzl:plaidml.bzl", "plaidml_cc_test")

plaidml_cc_test(
    name = "test",
    srcs = glob(["*.cc"]),
    deps = [
        "//tile/codegen",
        "//tile/lang",
    ],
)
