# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_test(
    name = "plaid_ir_test",
    srcs = ["plaid_ir_test.cc"],
    tags = ["llvm"],
    deps = ["@mlir//:ir"],
)
