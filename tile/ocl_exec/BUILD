package(default_visibility = ["//visibility:public"])

load("//bzl:conda.bzl", "conda_binary")
load("//bzl:plaidml.bzl", "plaidml_cc_binary")

plaidml_cc_binary(
    tags = ["manual"],
    name = "ocl_exec",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    deps = [
        "@boost//:filesystem",
        "//base/config",
        "//tile/base",
        "//tile/codegen",
        "//tile/lang",
        "//tile/lib",
        "//tile/util",
    ],
)

conda_binary(
    tags = ["manual"],
    name = "test",
    srcs = ["ocl_exec_test.py"],
    data = ["gpu.json.j2"],
    env = "@com_intel_plaidml_conda_ocl_exec//:env",
    deps = [":ocl_exec"],
    main = "ocl_exec_test.py",
)

