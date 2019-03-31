# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:conda.bzl", "conda_binary")

conda_binary(
    name = "gencfg",
    srcs = ["gencfg.py"],
    env = "@com_intel_plaidml_conda_pmlc//:env",
    main = "gencfg.py",
    deps = ["//tile/codegen:proto_py"],
)
