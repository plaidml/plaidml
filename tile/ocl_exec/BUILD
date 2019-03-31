package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "ocl_exec",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    deps = [
        "//tile/base",
        "//tile/codegen",
        "//tile/lang",
        "//tile/targets/gpu",
        "//tile/util",
        "@boost//:filesystem",
    ] + select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": ["//tile/targets/cpu"],
    }),
)
