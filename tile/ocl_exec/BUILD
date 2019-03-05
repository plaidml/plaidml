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
  outs = ["gpu.json"],
  srcs = ["gpu.json.j2"],
  tools = [":make_config"],
  cmd = "$(location :make_config) $(location :gpu.json.j2) > $@",
)

plaidml_cc_library(
    name = "ocl_exec",
    data = [":gpu_config"],
    srcs = [
        "emitsem.h",
	"emitsem.cc",
	"stripe_gen.h",
	"stripe_gen.cc",
    ],
    deps = [
        "@boost//:filesystem",
        "//base/config",
        "//tile/base",
	"//tile/codegen:codegen_base",
        "//tile/lang",
        "//tile/lib",
        "//tile/util",
    ],
)

plaidml_cc_binary(
    tags = ["manual"],
    name = "ocl_exec_bin",
    srcs = glob([
        "emitsem.h",
	"emitsem.cc",
	"ocl_exec.cc",
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
    deps = [":ocl_exec_bin"],
    main = "ocl_exec_test.py",
)

