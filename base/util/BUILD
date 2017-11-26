# Copyright 2017, Vertex.AI.
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "util_cc",
    srcs = [
        "env.cc",
        "error.cc",
        "json_transfer.cc",
        "logging.cc",
        "perf_counter.cc",
        "printstring.cc",
        "uuid.cc",
    ],
    hdrs = [
        "any_factory.h",
        "any_factory_map.h",
        "catch.h",
        "compat.h",
        "env.h",
        "error.h",
        "intern.h",
        "iterator_util.h",
        "json_transfer.h",
        "logging.h",
        "pdebug.h",
        "perf_counter.h",
        "printstring.h",
        "sync.h",
        "transfer_object.h",
        "type_url.h",
        "uuid.cc",
        "uuid.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@boost_archive//:boost",
        "@easylogging_repo//:easylogging",
        "@gflags_repo//:gflags",
        "@jsoncpp_git//:jsoncpp",
    ],
)
