load("@com_intel_plaidml//bzl:plaidml.bzl", "plaidml_cc_test")

plaidml_cc_test(
    name = "test",
    srcs = glob(["*.cc"]),
    deps = [
        "//testing:matchers",
        "//tile/codegen",
        "//tile/lang",
    ],
)
