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
        "//plaidml",
        "@openvino//:inference_engine",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "libplaidml-plugin.so",
    linkopts = PLAIDML_LINKOPTS,
    linkshared = 1,
    tags = TAGS,
    deps = [":lib"],
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
