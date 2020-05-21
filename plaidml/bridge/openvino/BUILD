load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_shlib")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "lib",
    srcs = glob([
        "src/plaidml_plugin/**/*.cpp",
        "src/plaidml_plugin/**/*.hpp",
    ]),
    hdrs = glob(["include/plaidml/*.hpp"]),
    copts = ["-Wno-pessimizing-move"],
    includes = ["src/plaidml_plugin"],
    deps = [
        "//plaidml",
        "@openvino//:inference_engine",
    ],
    alwayslink = 1,
)

plaidml_cc_shlib(
    name = "shlib",
    shlib_name = "plaidml-plugin",
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
