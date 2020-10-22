# Copyright 2020 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_test")
load("//pmlc:lit.bzl", "glob_lit_tests")

glob_lit_tests()

plaidml_cc_test(
    name = "openmp_test",
    srcs = ["openmp_test.cc"],
    copts = ["-w"],
    deps = ["@llvm-project//openmp"],
)
