package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license

load("@com_intel_plaidml//bzl:template.bzl", "template_rule")

cc_binary(
    name = "benchmark_app",
    srcs = glob([
        "inference-engine/samples/benchmark_app/**/*.hpp",
        "inference-engine/samples/benchmark_app/**/*.cpp",
    ]),
    data = [":plugins"],
    linkstatic = 0,
    deps = [
        ":inference_engine",
        ":mkldnn_plugin",
    ],
)

cc_library(
    name = "testing",
    srcs = glob([
        "inference-engine/tests/helpers/*common*cpp",
    ]),
    hdrs = glob([
        "inference-engine/tests/helpers/*common*hpp",
    ]),
    defines = [
        "DATA_PATH=NULL",
    ],
    includes = [
        "inference-engine/tests/helpers",
    ],
    deps = [
        ":inference_engine",
    ],
)

cc_library(
    name = "smoke_tests",
    srcs = [
        "inference-engine/tests/unit/engines/mkldnn/dump_test.cpp",
    ],
    hdrs = [
        "inference-engine/tests/unit/engines/mkldnn/graph/test_graph.hpp",
    ],
    data = [":plugins"],
    includes = [
        "inference-engine/tests/unit/engines/mkldnn/graph",
    ],
    deps = [
        ":mkldnn_plugin",
        ":testing",
        "@gmock//:gtest",
    ],
)

genrule(
    name = "plugins",
    outs = ["plugins.xml"],
    cmd = "echo \"<ie><plugins><plugin name=\\\"CPU\\\" location=\\\"./libmkldnn.so\\\"></plugin></plugins></ie>\" > $@",
)

template_rule(
    name = "mkldnn_version",
    src = "inference-engine/thirdparty/mkl-dnn/include/mkldnn_version.h.in",
    out = "inference-engine/thirdparty/mkl-dnn/include/mkldnn_version.h",
    substitutions = {
        "@MKLDNN_VERSION_MAJOR@": "1",
        "@MKLDNN_VERSION_MINOR@": "1",
        "@MKLDNN_VERSION_PATCH@": "1",
        "@MKLDNN_VERSION_HASH@": "afd",
    },
)

cc_library(
    name = "mkldnn_plugin",
    srcs = glob(
        [
            "inference-engine/thirdparty/mkl-dnn/src/**/*pp",
            "inference-engine/src/mkldnn_plugin/**/*pp",
        ],
        exclude = [
            "inference-engine/src/mkldnn_plugin/mkldnn/os/**/*.cpp",
            "inference-engine/src/mkldnn_plugin/nodes/ext_convert.cpp",
        ],
    ) + select({
        "@bazel_tools//src/conditions:windows": glob([
            "inference-engine/src/mkldnn_plugin/mkldnn/os/win/*.cpp",
        ]),
        "//conditions:default": glob([
            "inference-engine/src/mkldnn_plugin/mkldnn/os/lin/*.cpp",
        ]),
    }),
    hdrs = glob([
        "inference-engine/thirdparty/mkl-dnn/include/*",
        "inference-engine/thirdparty/mkl-dnn/src/common/*.hpp",
        "inference-engine/thirdparty/mkl-dnn/src/cpu/**/*.h*",
        "inference-engine/thirdparty/mkl-dnn/src/*.hpp",
    ]) + [":mkldnn_version"],
    includes = [
        "inference-engine/src/mkldnn_plugin/",
        "inference-engine/src/mkldnn_plugin/mkldnn",
        "inference-engine/thirdparty/mkl-dnn/include",
        "inference-engine/thirdparty/mkl-dnn/src",
        "inference-engine/thirdparty/mkl-dnn/src/common",
        "inference-engine/thirdparty/mkl-dnn/src/cpu",
    ],
    local_defines = [
        "COMPILED_CPU_MKLDNN_QUANTIZE_NODE",
    ],
    deps = [":inference_engine"],
    alwayslink = 1,
)

cc_library(
    name = "inference_engine",
    srcs = glob(
        [
            "inference-engine/src/extension/*pp",
            "inference-engine/src/extension/**/*pp",
            "inference-engine/src/extension/common/*",
            "inference-engine/src/inference_engine/*.cpp",
            "inference-engine/src/inference_engine/builders/*pp",
            "inference-engine/src/inference_engine/dumper/*.cpp",
            "inference-engine/src/inference_engine/ngraph_ops/*pp",
            "inference-engine/src/inference_engine/cpp_interfaces/**/*pp",
            "inference-engine/src/inference_engine/low_precision_transformations/**/*pp",
            "inference-engine/src/inference_engine/transform/**/*pp",
            "inference-engine/src/inference_engine/shape_infer/**/*pp",
            "inference-engine/src/preprocessing/*.cpp",
            "inference-engine/samples/*.cpp",
            "inference-engine/samples/common/**/*.cpp",
        ],
    ),
    hdrs = glob(
        [
            "inference-engine/src/extension/*.hpp",
            "inference-engine/src/extension/common/*",
            "inference-engine/include/**/*.h*",
            "inference-engine/samples/*.hpp",
            "inference-engine/samples/common/**/*.h*",
            "inference-engine/src/inference_engine/*.h",
            "inference-engine/src/inference_engine/cpp_interfaces/**/*.h*",
            "inference-engine/src/inference_engine/low_precision_transformations/**/*.h*",
            "inference-engine/src/inference_engine/shape_infer/**/.h*",
            "inference-engine/src/preprocessing/*.h*",
        ],
    ),
    defines = [
        "CI_BUILD_NUMBER=\\\"33\\\"",
        "IE_BUILD_POSTFIX=\\\"pml\\\"",
    ],
    includes = [
        "inference-engine/",
        "inference-engine/include/",
        "inference-engine/samples/common/",
        "inference-engine/samples/common/format_reader",
        "inference-engine/src/dumper/",
        "inference-engine/src/extension/",
        "inference-engine/src/extension/common",
        "inference-engine/src/inference_engine/",
        "inference-engine/src/preprocessing/",
    ],
    deps = [
        ":gapi",
        ":pugixml",
        "@gflags",
        "@gmock//:gtest",
        "@ngraph//:core",
        "@tbb",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gapi",
    srcs = glob(
        [
            "inference-engine/thirdparty/fluid/modules/gapi/src/api/*.cpp",
            "inference-engine/thirdparty/fluid/modules/gapi/src/compiler/**/*.cpp",
            "inference-engine/thirdparty/fluid/modules/gapi/src/executor/*.cpp",
            "inference-engine/thirdparty/fluid/modules/gapi/src/backends/common/*.cpp",
            "inference-engine/thirdparty/fluid/modules/gapi/src/backends/fluid/*.cpp",
            "inference-engine/thirdparty/fluid/modules/gapi/src/*.hpp",
        ],
        exclude = [
            "inference-engine/thirdparty/fluid/modules/gapi/src/api/operators.cpp",
            "inference-engine/thirdparty/fluid/modules/gapi/src/api/kernels_core.cpp",
            "inference-engine/thirdparty/fluid/modules/gapi/src/api/kernels_imgproc.cpp",
            "inference-engine/thirdparty/fluid/modules/gapi/src/api/render.cpp",
        ],
    ),
    hdrs = glob([
        "inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/*.hpp",
        "inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/gapi/*.hpp",
    ]),
    defines = [
        "GAPI_STANDALONE",
    ],
    includes = [
        "inference-engine/thirdparty/fluid/modules/gapi/include",
        "inference-engine/thirdparty/fluid/modules/gapi/src",
    ],
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
    includes = [
        "inference-engine/thirdparty/pugixml/src",
    ],
    strip_include_prefix = "inference-engine/thirdparty/pugixml/src",
)
