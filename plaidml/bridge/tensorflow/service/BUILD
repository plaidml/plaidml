load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    deps = [
        "//plaidml",
        "@org_tensorflow//tensorflow/compiler/xla/service:compiler",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
