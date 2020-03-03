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
    deps = [
        "inference_engine",
        "mkldnn_plugin",
    ],
    linkstatic = False,
)

genrule(
    name = "plugins",
    outs = ["plugins.xml"],
    cmd = 'echo "<ie><plugins><plugin name=\\"CPU\\" location=\\"./libmkldnn.so\\"></plugin></plugins></ie>" > $@',
)

template_rule(
    name = "mkldnn_version",
    substitutions = {
        "@MKLDNN_VERSION_MAJOR@": "1",
        "@MKLDNN_VERSION_MINOR@": "1",
        "@MKLDNN_VERSION_PATCH@": "1",
        "@MKLDNN_VERSION_HASH@": "afd",
    },
    src = "inference-engine/thirdparty/mkl-dnn/include/mkldnn_version.h.in",
    out = "inference-engine/thirdparty/mkl-dnn/include/mkldnn_version.h",
)

cc_library(
    name = "mkldnn_plugin",
    alwayslink = True,
    srcs = glob([
        #"inference-engine/thirdparty/mkl-dnn/src/*pp",
        "inference-engine/thirdparty/mkl-dnn/src/**/*pp",
        #"inference-engine/src/mkldnn_plugin/*pp",
        "inference-engine/src/mkldnn_plugin/**/*pp",
    ], exclude=[
        "inference-engine/src/mkldnn_plugin/**/win/*",
        "inference-engine/src/mkldnn_plugin/**/lin/*",
        "inference-engine/src/mkldnn_plugin/nodes/ext_convert.cpp"
    ]),
    hdrs = glob([
        "inference-engine/thirdparty/mkl-dnn/include/*",
        "inference-engine/thirdparty/mkl-dnn/src/common/*.hpp",
        #"inference-engine/thirdparty/mkl-dnn/src/cpu/*.hpp",
        "inference-engine/thirdparty/mkl-dnn/src/cpu/**/*.h*",
        "inference-engine/thirdparty/mkl-dnn/src/*.hpp",
        "inference-engine/src/mkldnn_plugin/**/*h*",
    ]) + [":mkldnn_version"],
    includes = [
        "inference-engine/thirdparty/mkl-dnn/include",
        "inference-engine/thirdparty/mkl-dnn/src/cpu",
        "inference-engine/thirdparty/mkl-dnn/src/common",
        "inference-engine/thirdparty/mkl-dnn/src",
        "inference-engine/src/mkldnn_plugin/",
        "inference-engine/src/mkldnn_plugin/mkldnn",
    ],
    defines = [
        "COMPILED_CPU_MKLDNN_QUANTIZE_NODE",
    ],
    deps = [ "inference_engine" ],
)

cc_library(
    name = "inference_engine",
    alwayslink = True,
    srcs = glob(
        [
            "inference-engine/src/extension/*pp",
            "inference-engine/src/extension/**/*pp",
            "inference-engine/src/extension/common/*",
            "inference-engine/src/inference_engine/*.cpp",
            "inference-engine/src/inference_engine/builders/*pp",
            "inference-engine/src/inference_engine/dumper/*.cpp",
            "inference-engine/src/inference_engine/ngraph_ops/*pp",
            "inference-engine/src/inference_engine/cpp_interfaces/*.*pp",
            "inference-engine/src/inference_engine/shape_infer/**/*.cpp",
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
            "inference-engine/src/inference_engine/shape_infer/**/.h*",
            "inference-engine/src/preprocessing/*.h*",
        ],
    ),
    includes = [
        "inference-engine/src/dumper/",
        "inference-engine/src/extension/",
        "inference-engine/src/extension/common",
        "inference-engine/src/inference_engine/",
        "inference-engine/src/preprocessing/",
        "inference-engine/include/",
        "inference-engine/",
        "inference-engine/samples/common/format_reader",
        "inference-engine/samples/common/",
    ],
    defines = [
        'CI_BUILD_NUMBER=\\"33\\"',
        'IE_BUILD_POSTFIX=\\"pml\\"',
        "GAPI_STANDALONE",
    ],
    deps = [
        "gapi",
        "@gmock//:gtest",
        "@gflags",
        "pugixml",
        "@ngraph//:core",
        "@tbb",
    ],
)

cc_library(
    name = "gapi",
    srcs = glob([
    ]),
    hdrs = glob([
        "inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/*.hpp",
        "inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/gapi/*.hpp",
    ]),
    includes = [
        "inference-engine/thirdparty/fluid/modules/gapi/include",
    ],
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
