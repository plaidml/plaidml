load("//bzl:conda.bzl", "conda_binary")
load("//bzl:plaidml.bzl", "plaidml_cc_binary")

conda_binary(
    name = "make_config",
    srcs = ["make_config.py"],
    env = "@com_intel_plaidml_conda_pmlc//:env",
    main = "make_config.py",
)

genrule(
    name = "cpu_config",
    srcs = ["//tile/targets/cpu:cpu.json.j2"],
    outs = ["cpu.json"],
    cmd = "$(location :make_config) $(location //tile/targets/cpu:cpu.json.j2) > $@",
    tools = [":make_config"],
)

plaidml_cc_binary(
    name = "cpu",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
    ),
    copts = [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
    ],
    data = [":cpu_config"],
    visibility = ["//visibility:public"],
    deps = [
        "//base/config",
        "//base/util:runfiles_db",
        "//tile/bilp",
        "//tile/codegen",
        "//tile/lang",
        "//tile/lib",
        "//tile/stripe",
    ],
)
