load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "shared_tests_instances",
    srcs = glob(["*.cpp"]),
    copts = select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": ["-Wno-deprecated-declarations"],
    }),
    tags = [
        "skip_macos",
        "skip_windows",
    ],
    data = [
        "//plaidml/bridge/openvino:libplaidml-plugin.so",
        "//:ov_plugins_xml",
    ],
    deps = [
        "//pmlc/util:logging",
        "@gflags",
        "@gmock//:gtest_main",
        "@openvino//:inference_engine",
        "@openvino//:common_test_utils",
        "@openvino//:shared_plugin_tests",
    ],
)
