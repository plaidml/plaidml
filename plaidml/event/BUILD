load("//bzl:plaidml.bzl", "plaidml_proto_library", "plaidml_py_library", "plaidml_py_binary")

plaidml_proto_library(
    name = "proto",
    srcs = ["event.proto"],
    visibility = ["//visibility:public"],
)

plaidml_py_library(
    name = "events_py",
    srcs = ["events.py"],
    visibility = ["//visibility:public"],
    deps = [":proto_py"],
)

plaidml_py_library(
    name = "filelog_py",
    srcs = ["filelog.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":events_py",
        ":proto_py",
    ],
)

plaidml_py_binary(
    name = "buffer_diff",
    srcs = ["buffer_diff.py"],
    visibility = ["//visibility:public"],
    deps = [":filelog_py"],
)
