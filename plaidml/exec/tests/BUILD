# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_test")

plaidml_cc_test(
    name = "list_devices_test",
    srcs = ["list_devices_test.cc"],
    deps = [
        "//plaidml/edsl",
        "//plaidml/exec",
        "//plaidml:runtimes",
    ],
    linkstatic = True,
    size = "small",
)
