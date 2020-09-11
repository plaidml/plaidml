load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@rules_cc//cc:defs.bzl", "cc_binary")
load("//bzl:plaidml.bzl", "PLAIDML_LINKOPTS", "plaidml_cc_library")

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

cc_binary(
    name = "benchmark_app",
    srcs = glob([
        "inference-engine/samples/benchmark_app/**/*.cpp",
    ]),
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": [
            "-pthread",
            "-lm",
            "-ldl",
        ],
    }),
    # linkstatic = 0,
    data = [
        ":plugins",
    ],
    deps = [
        "@openvino//:benchmark_app",
        ":libinference_engine_ir_reader.so",
        ":libplaidml-plugin.so",
    ],
)

cc_binary(
    name = "plaidml-plugin",
    linkopts = PLAIDML_LINKOPTS,
    linkshared = 1,
    tags = TAGS,
    deps = [
        ":lib",
        "//plaidml:runtimes",
        "//plaidml:targets",
    ],
)

# TODO
cc_binary(
    name = "libinference_engine_ir_reader.so",
    # linkopts = TODO,
    # linkopts = PLAIDML_LINKOPTS,
    linkshared = 1,
    tags = TAGS,
    deps = [
        "@openvino//:ie_reader",
        # "@openvino//:inference_engine",
    ],
)

plaidml_cc_library(
    name = "testenv",
    testonly = 1,
    srcs = ["testenv.cpp"],
    tags = TAGS,
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_googletest//:gtest",
        "@gflags",
        "@openvino//:shared_plugin_tests",
    ],
    alwayslink = 1,
)

write_file(
    name = "plugins",
    out = "plugins.xml",
    content = [
        "<ie>",
        "   <plugins>",
        "       <plugin name='PlaidML' location='plaidml/bridge/openvino/libplaidml-plugin.so'/>",
        "   </plugins>",
        "</ie>",
        "",
    ],
)
