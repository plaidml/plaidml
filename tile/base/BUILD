# Copyright 2018, Intel Corp.

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_proto_library")

plaidml_cc_library(
    name = "base",
    srcs = [
        "dbgsync.cc",
        "shape.cc",
        "validate.cc",
    ],
    hdrs = [
        "buffer.h",
        "dbgsync.h",
        "lru_cache.h",
        "namespaces.h",
        "platform.h",
        "program.h",
        "shape.h",
        "validate.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/context",
        "//base/util",
        "//tile/proto:shape_cc",
        "@boost",
        "@boost//:thread",
        "@easylogging",
        "@gflags",
    ],
)

plaidml_cc_library(
    name = "hal",
    hdrs = [
        "hal.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":schedule",
        "//base/context",
        "//tile/lang",
        "//tile/proto:hal_cc",
        "@boost//:thread",
    ],
)

plaidml_cc_library(
    name = "schedule",
    srcs = ["schedule.cc"],
    hdrs = ["schedule.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//base/context",
        "//base/util",
        "//tile/proto:schedule_cc",
    ],
)

plaidml_cc_library(
    name = "program_cache",
    srcs = ["program_cache.cc"],
    hdrs = ["program_cache.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        "//tile/lang",
        "//tile/proto:support",
    ],
)

plaidml_cc_library(
    name = "platform_test",
    testonly = True,
    srcs = ["platform_test.cc"],
    hdrs = ["platform_test.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        "//testing:matchers",
        "//tile/proto:support",
        "@gmock//:gtest",
        "@half",
    ],
    alwayslink = True,
)
