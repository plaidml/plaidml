load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//bzl:plaidml.bzl", "PLAIDML_LINKOPTS")

package(default_visibility = ["//visibility:public"])

TAGS = [
    "skip_macos",
    "skip_windows",
]

DATA = [
    "//plaidml/bridge/openvino:libplaidml-plugin.so",
    "//plaidml/bridge/openvino:plugins",
    "//:ov_plugins_xml",  # TODO
]

DEPS = [
    "//pmlc/util:logging",
    "@gflags",
    "@gmock//:gtest_main",
    "@openvino//:common_test_utils",
    # "@openvino//:helpers",  # TODO
    "@openvino//:inference_engine",
    "@openvino//:shared_plugin_tests",
]

cc_library(
    name = "lib",
    srcs = glob(
        ["*.cpp"],
        exclude = [
            # New Excludes (TODO)
            "batch_norm_inference.cpp",
            "broadcast.cpp",
            "convert_like.cpp",   # TODO: Broken not missing
            "divide.cpp",
            "floor_mod.cpp",
            "greater.cpp",   # TODO: Broken not missing
            "greater_equal.cpp",
            "less.cpp",
            "less_equal.cpp",
            "logical_and.cpp",
            "logical_not.cpp",
            "logical_or.cpp",
            "logical_xor.cpp",
            "lrn.cpp",   # TODO: Broken not missing
            "mat_mul.cpp",  # TODO : Broken not missing
            "maximum.cpp",
            "minimum.cpp",
            "mod.cpp",
            "mvn.cpp",  # TODO: Broken not missing
            "normalize_l2.cpp",
            "not_equal.cpp",
            "pad.cpp",
            "power.cpp",
            "reduce_logical_and.cpp",
            "reduce_logical_or.cpp",
            "reduce_max.cpp",
            "reduce_mean.cpp",
            "reduce_min.cpp",
            "reduce_prod.cpp",
            "reduce_sum.cpp",
            "reshape.cpp",  # TODO: Broken not missing
            "softmax.cpp",  # TODO: Broken not missing
            "space_to_depth.cpp",  # TODO: Broken not missing
            "squared_difference.cpp",
            "squeeze.cpp",
            "strided_slice.cpp",  # TODO: Broken not missing
            "tile.cpp",
            "transpose.cpp",   # TODO: Broken not missing
            "unsqueeze.cpp",
            # Original Excludes
            "cum_sum.cpp",
            "extract_image_patches.cpp",
            "space_to_batch.cpp",
        ],
    ),
    copts = ["-Wno-deprecated-declarations"],
    tags = TAGS,  # TODO
    deps = DEPS + [ "//plaidml/bridge/openvino/tests/functional/plugin/plaidml/shared_tests_instances" ],  # TODO: need addon?
    # data = DATA,  # TODO: Needed?
    # linkopts = PLAIDML_LINKOPTS,  # TODO: Needed?
)

cc_library(
    name = "smoke_lib",
    srcs = glob(
        ["*.cpp"],
        exclude = [
            # New Excludes (TODO)
            "batch_norm_inference.cpp",
            "broadcast.cpp",
            "convert_like.cpp",   # TODO: Broken not missing
            "divide.cpp",
            "floor_mod.cpp",
            "greater.cpp",   # TODO: Broken not missing
            "greater_equal.cpp",
            "less.cpp",
            "less_equal.cpp",
            "logical_and.cpp",
            "logical_not.cpp",
            "logical_or.cpp",
            "logical_xor.cpp",
            "lrn.cpp",   # TODO: Broken not missing
            "mat_mul.cpp",  # TODO : Broken not missing
            "maximum.cpp",
            "minimum.cpp",
            "mod.cpp",
            "mvn.cpp",  # TODO: Broken not missing
            "normalize_l2.cpp",
            "not_equal.cpp",
            "pad.cpp",
            "power.cpp",
            "reduce_logical_and.cpp",
            "reduce_logical_or.cpp",
            "reduce_max.cpp",
            "reduce_mean.cpp",
            "reduce_min.cpp",
            "reduce_prod.cpp",
            "reduce_sum.cpp",
            "reshape.cpp",  # TODO: Broken not missing
            "softmax.cpp",  # TODO: Broken not missing
            "space_to_depth.cpp",  # TODO: Broken not missing
            "squared_difference.cpp",
            "squeeze.cpp",
            "strided_slice.cpp",  # TODO: Broken not missing
            "tile.cpp",
            "transpose.cpp",   # TODO: Broken not missing
            "unsqueeze.cpp",
            # Exclusions from update
            "convert.cpp",                          # Known errors
            "depth_to_space.cpp",                   # Long  TODO: Make smoke test version
            # Original Excludes
            "convert_like.cpp",  # Known errors
            "convolution.cpp",  # Long
            "convolution_backprop_data.cpp",  # Long
            "cum_sum.cpp",  # Doesn't compile
            "equal.cpp",  # Known errors
            "extract_image_patches.cpp",  # Doesn't compile
            "fake_quantize.cpp",  # Known errors (intermittent)
            "floor_mod.cpp",  # Known errors
            "greater.cpp",  # Known errors
            "greater_equal.cpp",  # Known errors
            "group_convolution_backprop_data.cpp",  # Long
            "less.cpp",  # Known errors
            "less_equal.cpp",  # Known errors
            "logical_and.cpp",  # Known errors
            "logical_not.cpp",  # Known errors
            "logical_or.cpp",  # Known errors
            "logical_xor.cpp",  # Known errors
            "not_equal.cpp",  # Known errors
            "reduce_logical_and.cpp",  # Known errors
            "reduce_logical_or.cpp",  # Known errors
            "pooling.cpp",  # Long, known errors
            "select.cpp",  # Known errors
            "split.cpp",  # Known errors
            "space_to_batch.cpp",  # Doesn't compile
            "tile.cpp",  # Known errors
        ],
    ),
    copts = ["-Wno-deprecated-declarations"],
    tags = TAGS,  # TODO
    deps = DEPS,  # TODO
)

cc_test(
    name = "single_layer_tests",
    data = DATA,
    linkopts = PLAIDML_LINKOPTS,
    tags = TAGS + ["manual"],
    deps = [":lib"],
)

# Tests suitable for CI, skips flakey, broken, and long tests
cc_test(
    name = "smoke",
    data = DATA,
    linkopts = PLAIDML_LINKOPTS,
    tags = TAGS,
    deps = [":smoke_lib"],
)
