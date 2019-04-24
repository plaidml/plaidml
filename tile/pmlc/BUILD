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

py_binary(
    name = "pmlc_py",
    srcs = ["pmlc.py"],
    data = [":pmlc_bin"],
    legacy_create_init = False,
    main = "pmlc.py",
)

plaidml_cc_library(
    name = "lib",
    srcs = ["pmlc.cc"],
    hdrs = ["pmlc.h"],
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
    ] + select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": [
            "//tile/targets/cpu",
            "//tile/targets/gpu",
        ],
    }),
)

plaidml_cc_binary(
    name = "pmlc_bin",
    srcs = ["main.cc"],
    deps = [":lib"],
)
