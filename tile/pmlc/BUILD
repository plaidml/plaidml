package(default_visibility = ["//visibility:public"])

load("//bzl:conda.bzl", "conda_binary")
load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

conda_binary(
    name = "pmlc",
    srcs = ["pmlc.py"],
    env = "@com_intel_plaidml_conda_pmlc//:env",
    main = "pmlc.py",
    deps = [":pmlc_bin"],
)

plaidml_cc_binary(
    name = "pmlc_bin",
    srcs = glob([
        "*.h",
        "*.cc",
    ]),
    deps = [
        "//base/config",
        "//tile/base",
        "//tile/codegen:base",
        "//tile/lang",
        "//tile/lib",
        "//tile/util",
    ],
)
