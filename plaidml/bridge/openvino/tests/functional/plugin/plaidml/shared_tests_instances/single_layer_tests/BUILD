load("@rules_cc//cc:defs.bzl", "cc_test")
load("//bzl:plaidml.bzl", "PLAIDML_LINKOPTS")

package(default_visibility = ["//visibility:public"])

cc_test(
    name = "single_layer_tests",
    srcs = glob(
        ["*.cpp"],
        exclude = [
            "cum_sum.cpp",
            "extract_image_patches.cpp",
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

# Tests suitable for CI, skips flakey, broken, and long tests
cc_test(
    name = "smoke",
    srcs = glob(
        ["*.cpp"],
        exclude = [
            "convert_like.cpp",                     # Known errors
            "convolution.cpp",                      # Long
            "convolution_backprop_data.cpp",        # Long
            "cum_sum.cpp",                          # Doesn't compile
            "equal.cpp",                            # Known errors
            "extract_image_patches.cpp",            # Doesn't compile
            "fake_quantize.cpp",                    # Known errors (intermittent)
            "floor_mod.cpp",                        # Known errors
            "greater.cpp",                          # Known errors
            "greater_equal.cpp",                    # Known errors
            "group_convolution_backprop_data.cpp",  # Long
            "less.cpp",                             # Known errors
            "less_equal.cpp",                       # Known errors
            "logical_and.cpp",                      # Known errors
            "logical_not.cpp",                      # Known errors
            "logical_or.cpp",                       # Known errors
            "logical_xor.cpp",                      # Known errors
            "not_equal.cpp",                        # Known errors
            "reduce_logical_and.cpp",               # Known errors
            "reduce_logical_or.cpp",                # Known errors
            "mvn.cpp",                              # Known errors
            "pooling.cpp",                          # Long, known errors
            "select.cpp",                           # Known errors
            "split.cpp",                            # Known errors
            "space_to_batch.cpp",                   # Doesn't compile
            "tile.cpp",                             # Known errors
        ],
    ),
    copts = ["-Wno-deprecated-declarations"],
    data = [
        "//plaidml/bridge/openvino:libplaidml-plugin.so",
        "//plaidml/bridge/openvino:plugins",
    ],
    linkopts = PLAIDML_LINKOPTS,
    tags = [
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
