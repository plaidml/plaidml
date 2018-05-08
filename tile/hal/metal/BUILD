load(
    "@vertexai_plaidml//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
)

plaidml_cc_library(
    name = "metal",
    srcs = [
        "emit.cc",
        "hal.cc",
    ],
    hdrs = ["hal.h"],
    tags = ["metal"],
    visibility = ["//visibility:public"],
    deps = [
        "//tile/base:hal",
        "//tile/hal/opencl:proto_cc",
        "@mtlpp_repo//:mtlpp_cc",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "platform_test",
    srcs = ["platform_test.cc"],
    tags = ["metal"],
    deps = [
        ":metal",
        "//tile/base:platform_test",
        "//tile/platform/local_machine",
    ],
)
