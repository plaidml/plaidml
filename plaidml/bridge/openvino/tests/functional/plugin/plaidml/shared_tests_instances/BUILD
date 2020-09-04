load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "shared_tests_instances",
    srcs = glob(["*.cpp"]),
    copts = select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": ["-Wno-deprecated-declarations"],
    }),
    data = [
        "//:ov_plugins_xml",
        "//plaidml/bridge/openvino:plaidml-plugin",
    ],
    tags = [
        "skip_macos",
        "skip_windows",
    ],
    deps = [
        "//pmlc/util:logging",
        "@gflags",
        "@gmock//:gtest_main",
        "@openvino//:common_test_utils",
        "@openvino//:inference_engine",
        "@openvino//:shared_plugin_tests",
    ],
)
