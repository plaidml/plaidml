load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    deps = [
        "//plaidml",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/compiler/xla:xla_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
