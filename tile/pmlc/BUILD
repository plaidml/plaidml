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
    defines = select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": ["ENABLE_LLVM_BITCODE"],
    }),
    deps = [
        "//base/config",
        "//tile/base",
        "//tile/codegen",
        "//tile/lang",
        "//tile/lib",
        "//tile/util",
        "//tile/targets/gpu",
    ] + select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": ["//tile/targets/cpu"],
    }),
)
