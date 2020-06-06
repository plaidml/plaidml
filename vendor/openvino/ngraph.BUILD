load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ngraph",
    srcs = glob(
        [
            "src/ngraph/*.cpp",
            "src/ngraph/*.hpp",
            "src/ngraph/autodiff/*.cpp",
            "src/ngraph/builder/*.cpp",
            "src/ngraph/descriptor/**/*.cpp",
            "src/ngraph/distributed/*.cpp",
            "src/ngraph/op/**/*.cpp",
            "src/ngraph/opsets/*.cpp",
            "src/ngraph/pass/*.cpp",
            "src/ngraph/pattern/**/*.cpp",
            "src/ngraph/runtime/*.cpp",
            "src/ngraph/runtime/dynamic/*.cpp",
            "src/ngraph/runtime/interpreter/*.cpp",
            "src/ngraph/state/*.cpp",
            "src/ngraph/type/*.cpp",
        ],
        exclude = [
            "src/ngraph/serializer.cpp",
        ],
    ),
    hdrs = glob([
        "src/ngraph/*.hpp",
        "src/ngraph/autodiff/*.hpp",
        "src/ngraph/builder/*.hpp",
        "src/ngraph/descriptor/**/*.hpp",
        "src/ngraph/distributed/*.hpp",
        "src/ngraph/op/*.hpp",
        "src/ngraph/op/**/*.hpp",
        "src/ngraph/opsets/*.hpp",
        "src/ngraph/pass/*.hpp",
        "src/ngraph/pattern/*.hpp",
        "src/ngraph/runtime/**/*.hpp",
        "src/ngraph/state/*.hpp",
        "src/ngraph/pattern/**/*.hpp",
        "src/ngraph/type/*.hpp",
    ]),
    defines = [
        "NGRAPH_JSON_DISABLE",
        "NGRAPH_VERSION=\\\"0.21.0\\\"",
    ],
    includes = [
        "src",
        "src/ngraph",
    ],
    local_defines = [
        "PROJECT_ROOT_DIR=\\\"./\\\"",
        "SHARED_LIB_PREFIX=\\\"\\\"",
        "SHARED_LIB_SUFFIX=\\\"\\\"",
    ],
)
