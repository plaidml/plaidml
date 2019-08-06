# Copyright 2019 Intel Corporation.

load("//bzl:conda.bzl", "conda_test")

conda_test(
    name = "tests",
    srcs = ["lit.py"],
    data = glob([
        "*.mlir",
        "lit.cfg.py",
        "lit.site.cfg.py",
    ]) + [
        "//pmlc/tools/pmlc-opt",
        "@llvm//:lit",
        "@llvm//:FileCheck",
        "@llvm//:count",
        "@llvm//:not",
    ],
    env = "@com_intel_plaidml_conda//:env",
    main = "lit.py",
    tags = ["manual"],
)
