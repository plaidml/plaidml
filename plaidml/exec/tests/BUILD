# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_test")

plaidml_cc_test(
    name = "list_devices_test",
    size = "small",
    srcs = ["list_devices_test.cc"],
    linkstatic = True,
    deps = [
        "//plaidml/edsl",
        "//plaidml/exec",
        "//pmlc/rt:all_runtimes",
        "//pmlc/target:all_targets",
    ],
)
