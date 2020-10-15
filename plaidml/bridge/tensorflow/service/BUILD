load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    tags = ["skip_windows"],
    deps = [
        "//plaidml",
        "@org_tensorflow//tensorflow/compiler/xla/service:compiler",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "graph_util",
    srcs = ["graph_util.cc"],
    hdrs = ["graph_util.h"],
    tags = ["skip_windows"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/tf2xla:tf2xla",
        "@org_tensorflow//tensorflow/compiler/xla/client:client",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_parser",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
    ],
)
