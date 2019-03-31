# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:conda.bzl", "conda_binary")

conda_binary(
    name = "bin2h",
    srcs = ["bin2h.py"],
    env = "@com_intel_plaidml_conda_pmlc//:env",
    main = "bin2h.py",
)
