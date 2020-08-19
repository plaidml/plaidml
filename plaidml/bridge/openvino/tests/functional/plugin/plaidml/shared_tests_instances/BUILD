load("//bzl:plaidml.bzl", "PLAIDML_LINKOPTS", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

plaidml_cc_library(
    name = "shared_tests_instances",
    srcs = glob(["*.cpp"]),
    copts = ["-Wno-deprecated-declarations"],
    data = [  # TODO
        "//plaidml/bridge/openvino:libplaidml-plugin.so",
        "//:ov_plugins_xml",  # TODO
    ],
    linkopts = PLAIDML_LINKOPTS,  # TODO
    deps = [  # TODO
        "//pmlc/util:logging",
        "@gflags",
        "@gmock//:gtest_main",
        "@openvino//:inference_engine",
        "@openvino//:common_test_utils",
        "@openvino//:shared_plugin_tests",
    ],
)
