load("@rules_cc//cc:defs.bzl", "cc_library")
load("@com_intel_plaidml//bzl:template.bzl", "template_rule")

package(default_visibility = ["//visibility:public"])

TAGS = [
    "skip_macos",
    "skip_windows",
]

cc_library(
    name = "inc",
    hdrs = glob([
        "inference-engine/include/**/*.h",
        "inference-engine/include/**/*.hpp",
    ]),
    defines = [
        "CI_BUILD_NUMBER=\\\"0\\\"",
        "IE_BUILD_POSTFIX=\\\"\\\"",
    ],
    includes = [
        "inference-engine/include",
    ],
    tags = TAGS,
)

cc_library(
    name = "shared_plugin_tests",
    srcs = glob(
        ["inference-engine/tests/functional/plugin/shared/src/single_layer_tests/*.cpp"],
        exclude = [
            "inference-engine/tests/functional/plugin/shared/src/single_layer_tests/ctc_greedy_decoder.cpp",
            "inference-engine/tests/functional/plugin/shared/src/single_layer_tests/cum_sum.cpp",
            "inference-engine/tests/functional/plugin/shared/src/single_layer_tests/extract_image_patches.cpp",
            "inference-engine/tests/functional/plugin/shared/src/single_layer_tests/nonzero.cpp",
            "inference-engine/tests/functional/plugin/shared/src/single_layer_tests/prior_box_clustered.cpp",
            "inference-engine/tests/functional/plugin/shared/src/single_layer_tests/proposal.cpp",
        ],
    ),
    hdrs = glob(
        ["inference-engine/tests/functional/plugin/shared/include/single_layer_tests/*.hpp"],
        exclude = [
            "inference-engine/tests/functional/plugin/shared/include/single_layer_tests/ctc_greedy_decoder.hpp",
            "inference-engine/tests/functional/plugin/shared/include/single_layer_tests/cum_sum.hpp",
            "inference-engine/tests/functional/plugin/shared/include/single_layer_tests/extract_image_patches.hpp",
            "inference-engine/tests/functional/plugin/shared/include/single_layer_tests/nonzero.hpp",
            "inference-engine/tests/functional/plugin/shared/include/single_layer_tests/prior_box_clustered.hpp",
            "inference-engine/tests/functional/plugin/shared/include/single_layer_tests/proposal.hpp",
        ],
    ),
    includes = ["inference-engine/tests/functional/plugin/shared/include"],
    deps = [
        ":functional_test_utils",
        ":inference_engine",
        ":ngraph_function_tests",
        "@gmock//:gtest",
    ],
)

cc_library(
    name = "functional_test_utils",
    srcs = glob(["inference-engine/tests/ie_test_utils/functional_test_utils/*.cpp"]),
    hdrs = glob(
        ["inference-engine/tests/ie_test_utils/functional_test_utils/*.hpp"],
        # exclude = ["inference-engine/tests/ie_test_utils/functional_test_utils/skip_tests_config.hpp"],  # TODO
    ),
    includes = ["inference-engine/tests/ie_test_utils"],
    deps = [
        "ngraph_function_tests",  # TODO: Ok?
        ":inference_engine",
        "@gmock//:gtest",
    ],
)

cc_library(
    name = "ngraph_function_tests",
    srcs = glob([
        "inference-engine/tests/ngraph_functions/src/*.cpp",
        "inference-engine/tests/ngraph_functions/src/utils/*.cpp",
    ]),
    hdrs = glob([
        "inference-engine/tests/ngraph_functions/include/ngraph_functions/*.hpp",
        "inference-engine/tests/ngraph_functions/include/ngraph_functions/utils/*.hpp",
    ]),
    includes = [
        "inference-engine/tests/ngraph_functions/include/",
    ],
    deps = [
        ":inference_engine",
    ],
)

cc_library(
    name = "common_test_utils",
    srcs = glob(["inference-engine/tests/ie_test_utils/common_test_utils/*.cpp"]),
    hdrs = glob(["inference-engine/tests/ie_test_utils/common_test_utils/*.hpp"]),
    copts = ["-w"],
    includes = ["inference-engine/tests/ie_test_utils"],
    deps = [
        ":inference_engine",
        "@gmock//:gtest",
    ],
)

# template_rule(
#     name = "test_model_repo",
#     src = "inference-engine/tests_deprecated/helpers/test_model_repo.hpp.in",
#     out = "inference-engine/tests_deprecated/helpers/test_model_repo.hpp",
#     substitutions = {
#         "@MODELS_LST@": "",
#     },
# )

# TODO: Might be broken
cc_library(
    name = "helpers",
    srcs = glob(["inference-engine/tests_deprecated/helpers/*.cpp"]),
    hdrs = glob(["inference-engine/tests_deprecated/helpers/*.hpp"]),  # + [":test_model_repo"],
    copts = ["-w"],
    defines = ["DATA_PATH=NULL"],
    includes = ["inference-engine/tests_deprecated/helpers"],
    deps = [
        ":common_test_utils",
        ":inference_engine",
    ],
)

cc_library(
    name = "legacy_api",
    srcs = glob([
        "inference-engine/src/legacy_api/src/**/*.cpp",
        "inference-engine/src/legacy_api/src/**/*.hpp",
        "inference-engine/src/legacy_api/src/**/*.h",
        "inference-engine/src/inference_engine/cnn_network_ngraph_impl.hpp",  # TODO: Ok?
        "inference-engine/src/transformations/include/ngraph_ops/crop_ie.hpp",  # TODO: Ok?
        "inference-engine/src/transformations/include/ngraph_ops/convolution_ie.hpp",  # TODO: Ok?
    ]),
    hdrs = glob([
        "inference-engine/src/legacy_api/include/**/*.hpp",
        "inference-engine/src/inference_engine/*.hpp",  # TODO: Ok?
        "inference-engine/src/transformations/include/**/*.hpp",  # TODO: Ok?
        # Or: "inference-engine/src/transformations/include/ngraph_ops/*.hpp",
    ]),
    copts = [
        "-w",
        "-isystem external/openvino/inference-engine/src/legacy_api/src",
    ],
    includes = [
        "inference-engine/src/inference_engine",  # TODO: Ok?
        "inference-engine/src/legacy_api/include",
        "inference-engine/src/transformations/include",  # TODO: Ok?
    ],
    tags = TAGS,
    deps = [
        ":inc",
        ":ngraph",
        ":plugin_api",
        ":pugixml",
        "@tbb",
    ],
)

cc_library(
    name = "low_precision_transformations",
    srcs = glob(["inference-engine/src/low_precision_transformations/src/**/*.cpp"]),
    copts = ["-w"],
    includes = ["inference-engine/src/low_precision_transformations/include"],
    tags = TAGS,
    deps = [
        ":inc",
        ":legacy_api",
    ],
)

cc_library(
    name = "plugin_api",
    hdrs = glob([
        "inference-engine/src/plugin_api/**/*.h",
        "inference-engine/src/plugin_api/**/*.hpp",
    ]),
    includes = ["inference-engine/src/plugin_api"],
    tags = TAGS,
)

cc_library(
    name = "preprocessing",
    srcs = glob(["inference-engine/src/preprocessing/*.cpp"]),
    copts = ["-w"],
    includes = ["inference-engine/src/preprocessing"],
    tags = TAGS,
    deps = [
        ":fluid_gapi",
        ":inc",
        ":plugin_api",
        "@tbb",
    ],
)

cc_library(
    name = "transformations",
    srcs = glob(["inference-engine/src/transformations/src/**/*.cpp"]),
    copts = ["-w"],
    includes = ["inference-engine/src/transformations/include"],
    tags = TAGS,
    deps = [
        ":inc",
        ":ngraph",
    ],
)

cc_library(
    name = "inference_engine",
    srcs = glob([
        "inference-engine/src/inference_engine/*.cpp",
        "inference-engine/src/inference_engine/threading/*.cpp",
    ]) + select({
        "@bazel_tools//src/conditions:windows": glob([
            "inference-engine/src/inference_engine/os/win/*.cpp",
        ]),
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": glob([
            "inference-engine/src/inference_engine/os/lin/*.cpp",
        ]),
    }),
    hdrs = glob([
        "inference-engine/src/readers/ir_reader/ie_ir_version.hpp",  # TODO: Ok?
    ]),
    copts = ["-w"],
    includes = [
        "inference-engine/src/inference_engine",
        "inference-engine/src/readers/ir_reader",  # TODO: Sufficient?
        "inference-engine/src/readers/reader_api",  # TODO: Sufficient?
    ],
    local_defines = [
        "ENABLE_IR_READER",
    ],
    tags = TAGS,
    deps = [
        ":inc",
        ":legacy_api",
        ":low_precision_transformations",
        ":ngraph",
        ":plugin_api",
        ":preprocessing",
        ":pugixml",
        ":transformations",
        "@tbb",
    ],
    alwayslink = 1,
)

cc_library(
    name = "fluid_gapi",
    srcs = glob([
        "inference-engine/thirdparty/fluid/modules/gapi/src/api/g*.cpp",
        "inference-engine/thirdparty/fluid/modules/gapi/src/compiler/*.cpp",
        "inference-engine/thirdparty/fluid/modules/gapi/src/compiler/passes/*.cpp",
        "inference-engine/thirdparty/fluid/modules/gapi/src/executor/*.cpp",
        "inference-engine/thirdparty/fluid/modules/gapi/src/backends/common/*.cpp",
        "inference-engine/thirdparty/fluid/modules/gapi/src/backends/fluid/*.cpp",
        "inference-engine/thirdparty/fluid/modules/gapi/src/*.hpp",
    ]),
    hdrs = glob([
        "inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/**/*.hpp",
    ]),
    copts = ["-w"],
    defines = [
        "GAPI_STANDALONE",
    ],
    includes = [
        "inference-engine/thirdparty/fluid/modules/gapi/include",
        "inference-engine/thirdparty/fluid/modules/gapi/src",
    ],
    tags = TAGS,
    deps = ["@ade"],
)

cc_library(
    name = "pugixml",
    srcs = glob([
        "inference-engine/thirdparty/pugixml/src/*.cpp",
    ]),
    hdrs = glob([
        "inference-engine/thirdparty/pugixml/src/*.hpp",
    ]),
    copts = ["-w"],
    includes = [
        "inference-engine/thirdparty/pugixml/src",
    ],
    strip_include_prefix = "inference-engine/thirdparty/pugixml/src",
)

cc_library(
    name = "ngraph",
    srcs = glob(
        [
            "ngraph/src/ngraph/*.cpp",
            "ngraph/src/ngraph/*.hpp",
            "ngraph/src/ngraph/autodiff/*.cpp",
            "ngraph/src/ngraph/builder/*.cpp",
            "ngraph/src/ngraph/descriptor/**/*.cpp",
            "ngraph/src/ngraph/distributed/*.cpp",
            "ngraph/src/ngraph/op/*.cpp",  # TODO: Needed?
            "ngraph/src/ngraph/op/**/*.cpp",
            "ngraph/src/ngraph/opsets/*.cpp",
            "ngraph/src/ngraph/pass/*.cpp",
            "ngraph/src/ngraph/pattern/**/*.cpp",
            "ngraph/src/ngraph/runtime/*.cpp",
            "ngraph/src/ngraph/runtime/dynamic/*.cpp",
            "ngraph/src/ngraph/runtime/interpreter/*.cpp",
            "ngraph/src/ngraph/runtime/reference/*.cpp",  # TODO: Ok?
            "ngraph/src/ngraph/state/*.cpp",
            "ngraph/src/ngraph/type/*.cpp",
            "ngraph/test/runtime/*.cpp",  # TODO: Ok?
            "ngraph/test/runtime/**/*.cpp",  # TODO: Ok?
        ],
        exclude = [
            "ngraph/src/ngraph/serializer.cpp",
            "ngraph/test/runtime/ie/*.cpp",  # TODO: Ok?
        ],
    ),
    hdrs = glob([
        "ngraph/src/ngraph/*.hpp",
        "ngraph/src/ngraph/autodiff/*.hpp",
        "ngraph/src/ngraph/builder/*.hpp",
        "ngraph/src/ngraph/descriptor/**/*.hpp",
        "ngraph/src/ngraph/distributed/*.hpp",
        "ngraph/src/ngraph/op/*.hpp",
        "ngraph/src/ngraph/op/**/*.hpp",
        "ngraph/src/ngraph/opsets/*.hpp",
        "ngraph/src/ngraph/pass/*.hpp",
        "ngraph/src/ngraph/pattern/*.hpp",
        "ngraph/src/ngraph/runtime/**/*.hpp",
        "ngraph/src/ngraph/state/*.hpp",
        "ngraph/src/ngraph/pattern/**/*.hpp",
        "ngraph/src/ngraph/type/*.hpp",
        "ngraph/test/runtime/*.hpp",  # TODO: Ok?
        "ngraph/test/runtime/**/*.hpp",  # TODO: Ok?
        "ngraph/test/runtime/ie/*.hpp",  # TODO: Ok?
    ]),
    defines = [
        "NGRAPH_JSON_DISABLE",
        "NGRAPH_VERSION=\\\"0.21.0\\\"",
    ],
    includes = [
        "ngraph/src",
        "ngraph/src/ngraph",
        "ngraph/test/runtime",  # TODO: Ok?
    ],
    local_defines = [
        "PROJECT_ROOT_DIR=\\\"./\\\"",
        "SHARED_LIB_PREFIX=\\\"\\\"",
        "SHARED_LIB_SUFFIX=\\\"\\\"",
    ],
)
