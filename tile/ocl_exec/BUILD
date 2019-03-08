package(default_visibility = ["//visibility:public"])

load("//bzl:conda.bzl", "conda_binary")
load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

conda_binary(
    name = "make_config",
    srcs = ["make_config.py"],
    env = "@com_intel_plaidml_conda_ocl_exec//:env",
    main = "make_config.py",
)

genrule(
    name = "gpu_config",
    srcs = ["gpu.json.j2"],
    outs = ["gpu.json"],
    cmd = "$(location :make_config) $(location :gpu.json.j2) > $@",
    tools = [":make_config"],
)

plaidml_cc_library(
    name = "ocl_exec",
    srcs = [
        "emitsem.cc",
        "emitsem.h",
        "stripe_gen.cc",
        "stripe_gen.h",
    ],
    data = [":gpu_config"],
    deps = [
        "//base/config",
        "//tile/base",
        "//tile/codegen:base",
        "//tile/lang",
        "//tile/lib",
        "//tile/util",
        "@boost//:filesystem",
    ],
)

plaidml_cc_binary(
    name = "ocl_exec_bin",
    srcs = glob([
        "emitsem.h",
        "emitsem.cc",
        "ocl_exec.cc",
    ]),
    tags = ["manual"],
    deps = [
        "//base/config",
        "//tile/base",
        "//tile/codegen",
        "//tile/lang",
        "//tile/lib",
        "//tile/util",
        "@boost//:filesystem",
    ],
)

conda_binary(
    name = "test",
    srcs = ["ocl_exec_test.py"],
    data = ["gpu.json.j2"],
    env = "@com_intel_plaidml_conda_ocl_exec//:env",
    main = "ocl_exec_test.py",
    tags = ["manual"],
    deps = [":ocl_exec_bin"],
)
