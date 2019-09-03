# Copyright 2019 Intel Corporation.

load("@rules_python//python:defs.bzl", "py_test")

py_test(
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
    main = "lit.py",
    tags = ["manual"],
)
