load("//bzl:plaidml.bzl", "plaidml_cc_test")

plaidml_cc_test(
    name = "test",
    srcs = glob(["*.cc"]),
    tags = ["llvm"],
    deps = [
        "//tile/codegen",
        "//tile/lang",
        "//tile/targets/cpu",
    ],
)
