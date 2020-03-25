package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license

cc_library(
    name = "core",
    srcs = glob(
        [
            "src/ngraph/*pp",
            "src/ngraph/autodiff/*pp",
            "src/ngraph/builder/*pp",
            "src/ngraph/descriptor/*pp",
            "src/ngraph/descriptor/**/*pp",
            "src/ngraph/op/*pp",
            "src/ngraph/op/**/*pp",
            "src/ngraph/opsets/**/*pp",
            "src/ngraph/pass/*pp",
            "src/ngraph/pattern/*pp",
            "src/ngraph/pattern/**/*pp",
            "src/ngraph/runtime/*pp",
            "src/ngraph/runtime/dynamic/*pp",
            "src/ngraph/type/*pp",
        ],
        exclude = [
            "src/ngraph/plaidml/*",
            "src/ngraph/serializer.cpp",
        ],
    ),
    hdrs = glob([
        "src/ngraph/*.hpp",
        "src/ngraph/autodiff/*.hpp",
        "src/ngraph/builder/*.hpp",
        "src/ngraph/descriptor/*.hpp",
        "src/ngraph/descriptor/**/*.hpp",
        "src/ngraph/op/*.hpp",
        "src/ngraph/op/**/*.hpp",
        "src/ngraph/pass/*.hpp",
        "src/ngraph/pattern/*.hpp",
        "src/ngraph/runtime/*.hpp",
        "src/ngraph/runtime/**/*.hpp",
        "src/ngraph/pattern/**/*.hpp",
        "src/ngraph/type/*.hpp",
    ]),
    defines = [
        "NGRAPH_VERSION=\\\"0.21.0\\\"",
        "PROJECT_ROOT_DIR=\\\"./\\\"",
        "SHARED_LIB_PREFIX=\\\"\\\"",
        "SHARED_LIB_SUFFIX=\\\"\\\"",
    ],
    includes = [
        "src/",
        "src/ngraph",
    ],
    deps = [
        "@nlo_json//:json",
    ],
)
