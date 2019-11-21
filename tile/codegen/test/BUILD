load("//bzl:plaidml.bzl", "plaidml_cc_test")

plaidml_cc_test(
    name = "test",
    srcs = glob(["*.cc"]),
    deps = [
        "//base/proto",
        "//plaidml2:testenv_mlir",
        "//testing:matchers",
        "//tile/codegen",
        "//tile/lang",
        "//tile/lib",
        "//tile/ocl_exec",
        "@boost//:filesystem",
    ],
)
