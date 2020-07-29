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
]

DEPS = [
    "//pmlc/util:logging",
    "@gflags",
    "@gmock//:gtest_main",
    "@openvino//:common_test_utils",
    "@openvino//:helpers",
    "@openvino//:inference_engine",
    "@openvino//:shared_plugin_tests",
]

cc_library(
    name = "lib",
    srcs = glob(
        ["*.cpp"],
        exclude = [
            # New Excludes (TODO)
            "acos.cpp",
            "activation.cpp",   # TODO: Broken not missing
            "add.cpp",
            "asin.cpp",
            "atan.cpp",
            "batch_norm_inference.cpp",
            "broadcast.cpp",
            "ceiling.cpp",
            "convert.cpp",   # TODO: Broken not missing
            "convert_like.cpp",   # TODO: Broken not missing
            "cos.cpp",
            "cosh.cpp",
            "depth_to_space.cpp",   # TODO: Broken not missing
            "divide.cpp",
            "eltwise.cpp",   # TODO: Broken not missing
            "elu.cpp",
            "equal.cpp",   # TODO: Broken not missing
            "erf.cpp",
            "floor.cpp",
            "floor_mod.cpp",
            "greater.cpp",   # TODO: Broken not missing
            "greater_equal.cpp",
            "hard_sigmoid.cpp",
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
            "multiply.cpp",
            "mvn.cpp",  # TODO: Broken not missing
            "negative.cpp",
            "normalize_l2.cpp",
            "not_equal.cpp",
            "power.cpp",
            "prelu.cpp",
            "reduce_logical_and.cpp",
            "reduce_logical_or.cpp",
            "reduce_max.cpp",
            "reduce_mean.cpp",
            "reduce_min.cpp",
            "reduce_prod.cpp",
            "reduce_sum.cpp",
            "reshape.cpp",  # TODO: Broken not missing
            "selu.cpp",
            "sin.cpp",
            "sinh.cpp",
            "softmax.cpp",  # TODO: Broken not missing
            "space_to_depth.cpp",  # TODO: Broken not missing
            "sqrt.cpp",
            "squared_difference.cpp",
            "squeeze.cpp",
            "strided_slice.cpp",  # TODO: Broken not missing
            "subtract.cpp",
            "tan.cpp",
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
    tags = TAGS,
    deps = DEPS,
)

cc_library(
    name = "smoke_lib",
    srcs = glob(
        ["*.cpp"],
        exclude = [
            # New Excludes (TODO)
            "acos.cpp",
            "activation.cpp",   # TODO: Broken not missing
            "add.cpp",
            "asin.cpp",
            "atan.cpp",
            "batch_norm_inference.cpp",
            "broadcast.cpp",
            "ceiling.cpp",
            "convert.cpp",   # TODO: Broken not missing
            "convert_like.cpp",   # TODO: Broken not missing
            "cos.cpp",
            "cosh.cpp",
            "depth_to_space.cpp",   # TODO: Broken not missing
            "divide.cpp",
            "eltwise.cpp",   # TODO: Broken not missing
            "elu.cpp",
            "equal.cpp",   # TODO: Broken not missing
            "erf.cpp",
            "floor.cpp",
            "floor_mod.cpp",
            "greater.cpp",   # TODO: Broken not missing
            "greater_equal.cpp",
            "hard_sigmoid.cpp",
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
            "multiply.cpp",
            "mvn.cpp",  # TODO: Broken not missing
            "negative.cpp",
            "normalize_l2.cpp",
            "not_equal.cpp",
            "power.cpp",
            "prelu.cpp",
            "reduce_logical_and.cpp",
            "reduce_logical_or.cpp",
            "reduce_max.cpp",
            "reduce_mean.cpp",
            "reduce_min.cpp",
            "reduce_prod.cpp",
            "reduce_sum.cpp",
            "reshape.cpp",  # TODO: Broken not missing
            "selu.cpp",
            "sin.cpp",
            "sinh.cpp",
            "softmax.cpp",  # TODO: Broken not missing
            "space_to_depth.cpp",  # TODO: Broken not missing
            "sqrt.cpp",
            "squared_difference.cpp",
            "squeeze.cpp",
            "strided_slice.cpp",  # TODO: Broken not missing
            "subtract.cpp",
            "tan.cpp",
            "tile.cpp",
            "transpose.cpp",   # TODO: Broken not missing
            "unsqueeze",
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
    tags = TAGS,
    deps = DEPS,
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
