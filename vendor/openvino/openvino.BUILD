load("@rules_cc//cc:defs.bzl", "cc_library")
load("@com_intel_plaidml//bzl:template.bzl", "template_rule")

package(default_visibility = ["//visibility:public"])

TAGS = [
    "skip_macos",
    "skip_windows",
]

# cc_library(
#     name = "benchmark_lib",
#     srcs = glob([
#         "inference-engine/samples/benchmark_app/**/*.hpp",
#         "inference-engine/samples/benchmark_app/**/*.cpp",
#     ]),
#     copts = ["-w"],
#     deps = [
#         ":inference_engine",
#     ],
#     alwayslink = 1,
# )

cc_library(
    name = "inc",
    hdrs = glob([
        "inference-engine/include/**/*.h",
        "inference-engine/include/**/*.hpp",
    ]),
    defines = [
        'CI_BUILD_NUMBER=\\"0\\"',
        'IE_BUILD_POSTFIX=\\"\\"',
    ],
    includes = [
        "inference-engine/include",
    ],
    tags = TAGS,
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

template_rule(
    name = "test_model_repo",
    src = "inference-engine/tests_deprecated/helpers/test_model_repo.hpp.in",
    out = "inference-engine/tests_deprecated/helpers/test_model_repo.hpp",
    substitutions = {
        "@MODELS_LST@": "",
    },
)

cc_library(
    name = "helpers",
    srcs = glob(["inference-engine/tests_deprecated/helpers/*.cpp"]),
    hdrs = glob(["inference-engine/tests_deprecated/helpers/*.hpp"]) + [":test_model_repo"],
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
    ]),
    hdrs = glob([
        "inference-engine/src/legacy_api/include/**/*.hpp",
    ]),
    copts = [
        "-w",
        "-isystem external/openvino/inference-engine/src/legacy_api/src",
    ],
    includes = [
        "inference-engine/src/legacy_api/include",
    ],
    tags = TAGS,
    deps = [
        ":inc",
        ":plugin_api",
        ":pugixml",
        "@ngraph",
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
        "@ngraph",
    ],
)

# cc_library(
#     name = "smoke_tests",
#     srcs = [
#         "inference-engine/tests/unit/engines/mkldnn/dump_test.cpp",
#     ],
#     hdrs = [
#         "inference-engine/tests/unit/engines/mkldnn/graph/test_graph.hpp",
#     ],
#     copts = ["-w"],
#     data = [":plugins"],
#     includes = [
#         "inference-engine/tests/unit/engines/mkldnn/graph",
#     ],
#     deps = [
#         ":mkldnn_plugin",
#         ":helpers",
#         "@gmock//:gtest",
#     ],
# )

# genrule(
#     name = "plugins",
#     outs = ["plugins.xml"],
#     cmd = "echo \"<ie><plugins><plugin name=\\\"CPU\\\" location=\\\"./libmkldnn.so\\\"></plugin></plugins></ie>\" > $@",
# )

# template_rule(
#     name = "mkldnn_version",
#     src = "inference-engine/thirdparty/mkl-dnn/include/mkldnn_version.h.in",
#     out = "inference-engine/thirdparty/mkl-dnn/include/mkldnn_version.h",
#     substitutions = {
#         "@MKLDNN_VERSION_MAJOR@": "1",
#         "@MKLDNN_VERSION_MINOR@": "1",
#         "@MKLDNN_VERSION_PATCH@": "1",
#         "@MKLDNN_VERSION_HASH@": "afd",
#     },
# )

# cc_library(
#     name = "mkldnn_plugin",
#     srcs = glob(
#         [
#             "inference-engine/thirdparty/mkl-dnn/src/**/*pp",
#             "inference-engine/src/mkldnn_plugin/**/*pp",
#         ],
#         exclude = [
#             "inference-engine/src/mkldnn_plugin/mkldnn/os/**/*.cpp",
#             "inference-engine/src/mkldnn_plugin/nodes/ext_convert.cpp",
#         ],
#     ) + select({
#         "@bazel_tools//src/conditions:darwin_x86_64": [],
#         "@bazel_tools//src/conditions:windows": glob([
#             "inference-engine/src/mkldnn_plugin/mkldnn/os/win/*.cpp",
#         ]),
#         "//conditions:default": glob([
#             "inference-engine/src/mkldnn_plugin/mkldnn/os/lin/*.cpp",
#         ]),
#     }),
#     hdrs = glob([
#         "inference-engine/thirdparty/mkl-dnn/include/*",
#         "inference-engine/thirdparty/mkl-dnn/src/common/*.hpp",
#         "inference-engine/thirdparty/mkl-dnn/src/cpu/**/*.h*",
#         "inference-engine/thirdparty/mkl-dnn/src/*.hpp",
#     ]) + [":mkldnn_version"],
#     copts = ["-w"],
#     includes = [
#         "inference-engine/src/mkldnn_plugin/",
#         "inference-engine/src/mkldnn_plugin/mkldnn",
#         "inference-engine/thirdparty/mkl-dnn/include",
#         "inference-engine/thirdparty/mkl-dnn/src",
#         "inference-engine/thirdparty/mkl-dnn/src/common",
#         "inference-engine/thirdparty/mkl-dnn/src/cpu",
#     ],
#     local_defines = [
#         "COMPILED_CPU_MKLDNN_QUANTIZE_NODE",
#     ],
#     deps = [":inference_engine"],
#     alwayslink = 1,
# )

# cc_library(
#     name = "inference_engine",
#     srcs = glob(
#         [
#             "inference-engine/src/extension/*pp",
#             "inference-engine/src/extension/**/*pp",
#             "inference-engine/src/extension/common/*",
#             "inference-engine/src/inference_engine/*.cpp",
#             "inference-engine/src/inference_engine/builders/*pp",
#             "inference-engine/src/inference_engine/dumper/*.cpp",
#             "inference-engine/src/inference_engine/ngraph_ops/*pp",
#             "inference-engine/src/inference_engine/cpp_interfaces/**/*pp",
#             "inference-engine/src/inference_engine/low_precision_transformations/**/*pp",
#             "inference-engine/src/inference_engine/transform/**/*pp",
#             "inference-engine/src/inference_engine/shape_infer/**/*pp",
#             "inference-engine/src/preprocessing/*.cpp",
#             "inference-engine/samples/*.cpp",
#             "inference-engine/samples/common/**/*.cpp",
#         ],
#     ),
#     hdrs = glob(
#         [
#             "inference-engine/src/extension/*.hpp",
#             "inference-engine/src/extension/common/*",
#             "inference-engine/include/**/*.h*",
#             "inference-engine/samples/*.hpp",
#             "inference-engine/samples/common/**/*.h*",
#             "inference-engine/src/inference_engine/*.h",
#             "inference-engine/src/inference_engine/cpp_interfaces/**/*.h*",
#             "inference-engine/src/inference_engine/low_precision_transformations/**/*.h*",
#             "inference-engine/src/inference_engine/shape_infer/**/.h*",
#             "inference-engine/src/preprocessing/*.h*",
#         ],
#     ),
#     copts = ["-w"],
#     defines = [
#         "CI_BUILD_NUMBER=\\\"33\\\"",
#         "IE_BUILD_POSTFIX=\\\"pml\\\"",
#     ],
#     includes = [
#         "inference-engine/",
#         "inference-engine/include/",
#         "inference-engine/samples/common/",
#         "inference-engine/samples/common/format_reader",
#         "inference-engine/src/dumper/",
#         "inference-engine/src/extension/",
#         "inference-engine/src/extension/common",
#         "inference-engine/src/inference_engine/",
#         "inference-engine/src/preprocessing/",
#     ],
#     deps = [
#         ":gapi",
#         ":pugixml",
#         "@gflags",
#         "@gmock//:gtest",
#         "@ngraph//:core",
#         "@tbb",
#     ],
#     alwayslink = 1,
# )

cc_library(
    name = "inference_engine",
    srcs = glob([
        "inference-engine/src/inference_engine/*.cpp",
        # "inference-engine/src/inference_engine/builders/*.cpp",
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
    copts = ["-w"],
    includes = [
        "inference-engine/src/inference_engine",
    ],
    local_defines = [
        "ENABLE_IR_READER",
    ],
    tags = TAGS,
    deps = [
        ":inc",
        ":legacy_api",
        ":low_precision_transformations",
        ":plugin_api",
        ":preprocessing",
        ":pugixml",
        ":transformations",
        "@ngraph",
        "@tbb",
    ],
    alwayslink = 1,
)

# cc_library(
#     name = "fluid_gapi",
#     srcs = glob(
#         [
#             "inference-engine/thirdparty/fluid/modules/gapi/src/api/*.cpp",
#             "inference-engine/thirdparty/fluid/modules/gapi/src/compiler/**/*.cpp",
#             "inference-engine/thirdparty/fluid/modules/gapi/src/executor/*.cpp",
#             "inference-engine/thirdparty/fluid/modules/gapi/src/backends/common/*.cpp",
#             "inference-engine/thirdparty/fluid/modules/gapi/src/backends/fluid/*.cpp",
#             "inference-engine/thirdparty/fluid/modules/gapi/src/*.hpp",
#         ],
#         exclude = [
#             "inference-engine/thirdparty/fluid/modules/gapi/src/api/operators.cpp",
#             "inference-engine/thirdparty/fluid/modules/gapi/src/api/kernels_core.cpp",
#             "inference-engine/thirdparty/fluid/modules/gapi/src/api/kernels_imgproc.cpp",
#             "inference-engine/thirdparty/fluid/modules/gapi/src/api/render.cpp",
#         ],
#     ),
#     hdrs = glob([
#         "inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/*.hpp",
#         "inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/gapi/*.hpp",
#     ]),
#     copts = ["-w"],
#     defines = [
#         "GAPI_STANDALONE",
#     ],
#     includes = [
#         "inference-engine/thirdparty/fluid/modules/gapi/include",
#         "inference-engine/thirdparty/fluid/modules/gapi/src",
#     ],
#     deps = ["@ade"],
# )

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
