load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "stripejit",
    srcs = [
        "platform.cc",
        "platform.h",
        "program.cc",
        "program.h",
    ],
    hdrs = [
        "platform.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tile/base",
        "//tile/base:hal",
        "//tile/hal/util:selector",
        "//tile/hal/util:settings",
        "//tile/proto:support",
        "//tile/targets",
    ],
    alwayslink = 1,
)
