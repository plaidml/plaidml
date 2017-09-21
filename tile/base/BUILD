load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "base",
    srcs = [
        "dbgsync.cc",
        "validate.cc",
    ],
    hdrs = [
        "buffer.h",
        "dbgsync.h",
        "lru_cache.h",
        "namespaces.h",
        "platform.h",
        "program.h",
        "validate.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/context",
        "//base/util:util_cc",
        "//external:boost",
        "//external:boost_thread",
        "//external:easylogging",
        "//external:gflags",
        "//tile/proto:proto_cc",
    ],
)

plaidml_cc_library(
    name = "hal",
    hdrs = [
        "hal.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/context",
        "//external:boost_thread",
        "//tile/lang",
        "//tile/proto:hal_cc",
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
        "//external:gtest",
        "//testing:matchers",
    ],
    alwayslink = True,
)
