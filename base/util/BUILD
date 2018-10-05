# Copyright 2017-2018 Intel Corporation.
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_py_library")

plaidml_py_library(
    name = "py",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "util",
    srcs = [
        "env.cc",
        "error.cc",
        "hexdump.cc",
        "json_transfer.cc",
        "logging.cc",
        "perf_counter.cc",
        "printstring.cc",
        "uuid.cc",
        "zipfile.cc",
    ],
    hdrs = [
        "any_factory.h",
        "any_factory_map.h",
        "callback_map.h",
        "catch.h",
        "compat.h",
        "env.h",
        "error.h",
        "factory.h",
        "hexdump.h",
        "intern.h",
        "iterator_util.h",
        "json_transfer.h",
        "logging.h",
        "pdebug.h",
        "perf_counter.h",
        "printstring.h",
        "stream_container.h",
        "sync.h",
        "transfer_object.h",
        "type_url.h",
        "uuid.h",
        "zipfile.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@boost",
        "@easylogging",
        "@gflags",
        "@jsoncpp",
        "@minizip",
    ],
)

plaidml_cc_library(
    name = "runfiles_db",
    srcs = ["runfiles_db.cc"],
    hdrs = ["runfiles_db.h"],
    visibility = ["//visibility:public"],
    deps = [":util"],
)
