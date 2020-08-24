load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//bzl:plaidml.bzl", "PLAIDML_LINKOPTS")

package(default_visibility = ["//visibility:public"])

TAGS = [
    "skip_macos",
    "skip_windows",
]

DATA = [
    "//plaidml/bridge/openvino:libplaidml-plugin.so",
]

DEPS = [
    "//plaidml/bridge/openvino/tests/functional/plugin/plaidml/shared_tests_instances",
]

COPTS = select({
    "@bazel_tools//src/conditions:windows": [],
    "//conditions:default": ["-Wno-deprecated-declarations"],
})

# TODO: Pull forward the PlaidML-OpenVINO tests from the 2020.2 OV version to remove exclusions

cc_library(
    name = "lib",
    srcs = glob(
        ["*.cpp"],
        exclude = [
            # Exclusions for files lost in update to OV `master`
            "batch_norm_inference.cpp",
            "broadcast.cpp",
            "convert_like.cpp",   # Note: Broken not missing
            "divide.cpp",
            "floor_mod.cpp",
            "greater_equal.cpp",
            "less.cpp",
            "less_equal.cpp",
            "logical_and.cpp",
            "logical_not.cpp",
            "logical_or.cpp",
            "logical_xor.cpp",
            "mat_mul.cpp",  # Note : Broken not missing
            "maximum.cpp",
            "minimum.cpp",
            "mod.cpp",
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
            "squared_difference.cpp",
            "squeeze.cpp",
            "tile.cpp",
            "unsqueeze.cpp",
            # Original Excludes
            "cum_sum.cpp",
            "extract_image_patches.cpp",
            "space_to_batch.cpp",
        ],
    ),
    copts = COPTS,
    tags = TAGS,
    deps = DEPS,
)

cc_library(
    name = "smoke_lib",
    srcs = glob(
        ["*.cpp"],
        exclude = [
            # Exclusions for files lost in update to OV `master`
            "batch_norm_inference.cpp",
            "broadcast.cpp",
            "convert_like.cpp",   # Note: Broken not missing
            "divide.cpp",
            "floor_mod.cpp",
            "greater_equal.cpp",
            "less.cpp",
            "less_equal.cpp",
            "logical_and.cpp",
            "logical_not.cpp",
            "logical_or.cpp",
            "logical_xor.cpp",
            "mat_mul.cpp",  # Note : Broken not missing
            "maximum.cpp",
            "minimum.cpp",
            "mod.cpp",
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
            "squared_difference.cpp",
            "squeeze.cpp",
            "tile.cpp",
            "unsqueeze.cpp",
            # Exclusions from update
            "convert.cpp",  # Known errors
            "depth_to_space.cpp",  # Long
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
    copts = COPTS,
    tags = TAGS,
    deps = DEPS,
)

cc_test(
    name = "single_layer_tests",
    data = DATA,
    linkopts = PLAIDML_LINKOPTS,
    tags = TAGS + ["manual"],
    deps = [
        ":lib",
        "//plaidml/bridge/openvino:testenv",
    ],
)

# Tests suitable for CI, skips flakey, broken, and long tests
cc_test(
    name = "smoke",
    data = DATA,
    linkopts = PLAIDML_LINKOPTS,
    tags = TAGS,
    deps = [
        ":smoke_lib",
        "//plaidml/bridge/openvino:testenv",
    ],
)
