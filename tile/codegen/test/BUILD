load("//bzl:plaidml.bzl", "plaidml_cc_test")

plaidml_cc_test(
    name = "test",
    srcs = glob(["*.cc"]),
    deps = [
        "//base/proto",
        "//testing:matchers",
        "//tile/codegen",
        "//tile/lang",
        "//tile/lib",
        "//tile/ocl_exec",
        "@boost//:filesystem",
    ],
)
