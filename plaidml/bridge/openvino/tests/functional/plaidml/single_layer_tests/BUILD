load("@rules_cc//cc:defs.bzl", "cc_test")
load("//bzl:plaidml.bzl", "PLAIDML_LINKOPTS")

package(default_visibility = ["//visibility:public"])

TAGS = [
    "skip_macos",
    "skip_windows",
]

cc_test(
    name = "single_layer_tests",
    srcs = glob(["*.cpp"]),
    copts = ["-Wno-deprecated-declarations"],
    data = [
        "//plaidml/bridge/openvino:libplaidml-plugin.so",
        "//plaidml/bridge/openvino:plugins",
        "//:ov_plugins_xml",  # TODO
    ],
    linkopts = PLAIDML_LINKOPTS,
    tags = TAGS,
    deps = [
        "//pmlc/util:logging",
        "@gflags",
        "@openvino//:helpers",
        "@openvino//:inference_engine",
    ],
)
