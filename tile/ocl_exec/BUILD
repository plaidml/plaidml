package(default_visibility = ["//visibility:public"])

load("//bzl:conda.bzl", "conda_binary")
load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

conda_binary(
    name = "make_config",
    srcs = ["make_config.py"],
    env = "@com_intel_plaidml_conda_pmlc//:env",
    main = "make_config.py",
)

genrule(
    name = "gpu_config",
    srcs = ["//tile/targets/gpu:gpu.json.j2"],
    outs = ["gpu.json"],
    cmd = "$(location :make_config) $(location //tile/targets/gpu:gpu.json.j2) > $@",
    tools = [":make_config"],
)

plaidml_cc_library(
    name = "ocl_exec",
    srcs = [
        "emitsem.cc",
        "emitsem.h",
        "intrinsic.cc",
        "intrinsic.h",
        "stripe_gen.cc",
        "stripe_gen.h",
    ],
    data = [":gpu_config"],
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

plaidml_cc_binary(
    name = "bin",
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

py_test(
    name = "test",
    srcs = ["ocl_exec_test.py"],
    data = [
        ":bin",
        ":gpu_config",
    ],
    main = "ocl_exec_test.py",
    tags = ["manual"],
)
