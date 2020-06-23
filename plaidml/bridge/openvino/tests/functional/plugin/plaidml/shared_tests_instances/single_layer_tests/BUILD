load("@rules_cc//cc:defs.bzl", "cc_test")
load("//bzl:plaidml.bzl", "PLAIDML_LINKOPTS")

package(default_visibility = ["//visibility:public"])

cc_test(
    name = "single_layer_tests",
    srcs = glob(
        ["*.cpp"],
        exclude = [
            "convolution_backprop_data.cpp",
            "cum_sum.cpp",
            "extract_image_patches.cpp",
            "fake_quantize.cpp",
            "group_convolution.cpp",
            "group_convolution_backprop_data.cpp",
            "mvn.cpp",
            "select.cpp",
            "space_to_batch.cpp",
        ],
    ),
    copts = ["-Wno-deprecated-declarations"],
    data = [
        "//plaidml/bridge/openvino:libplaidml-plugin.so",
        "//plaidml/bridge/openvino:plugins",
    ],
    linkopts = PLAIDML_LINKOPTS,
    tags = [
        "manual",
        "skip_macos",
        "skip_windows",
    ],
    deps = [
        "//pmlc/util:logging",
        "@gflags",
        "@gmock//:gtest_main",
        "@openvino//:helpers",
        "@openvino//:inference_engine",
        "@openvino//:common_test_utils",
        "@openvino//:shared_plugin_tests",
    ],
)
