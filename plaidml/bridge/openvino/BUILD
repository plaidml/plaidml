load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

TAGS = [
    "skip_macos",
    "skip_windows",
]

plaidml_cc_library(
    name = "lib",
    srcs = glob([
        "src/plaidml_plugin/**/*.cpp",
        "src/plaidml_plugin/**/*.hpp",
    ]),
    hdrs = glob(["include/plaidml/*.hpp"]),
    copts = ["-Wno-pessimizing-move"],
    includes = ["src/plaidml_plugin"],
    tags = TAGS,
    deps = [
        "//plaidml:api",
        "@openvino//:inference_engine",
    ],
    alwayslink = 1,
)

plaidml_cc_binary(
    name = "plaidml-plugin",
    linkshared = 1,
    tags = TAGS,
    deps = [
        ":lib",
        "//plaidml:runtimes",
        "//plaidml:targets",
    ],
)

plaidml_cc_library(
    name = "testenv",
    testonly = 1,
    srcs = ["testenv.cpp"],
    tags = TAGS,
    deps = [
        "@com_google_googletest//:gtest",
        "@gflags",
        "@openvino//:shared_plugin_tests",
    ],
    alwayslink = 1,
)
