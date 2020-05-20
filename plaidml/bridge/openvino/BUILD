package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_test")

licenses(["notice"])  # BSD/MIT-like license

plaidml_cc_test(
    name = "smoke",
    tags = [
        "skip_windows",
    ],
    deps = [
        "@openvino//:smoke_tests",
    ],
)

cc_library(
    name = "plaidml-plugin",
    srcs = [
        "src/plaidml_plugin/plaidml_executable_network.cpp",
        "src/plaidml_plugin/plaidml_infer_request.cpp",
        "src/plaidml_plugin/plaidml_op.cpp",
        "src/plaidml_plugin/plaidml_op_clamp.cpp",
        "src/plaidml_plugin/plaidml_op_concat.cpp",
        #"src/plaidml_plugin/plaidml_op_convolution.cpp",
        "src/plaidml_plugin/plaidml_op_crop.cpp",
        "src/plaidml_plugin/plaidml_op_eltwise.cpp",
        "src/plaidml_plugin/plaidml_op_fc.cpp",
        "src/plaidml_plugin/plaidml_op_norm.cpp",
        "src/plaidml_plugin/plaidml_op_permute.cpp",
        #"src/plaidml_plugin/plaidml_op_pooling.cpp",
        "src/plaidml_plugin/plaidml_op_power.cpp",
        "src/plaidml_plugin/plaidml_op_relu.cpp",
        "src/plaidml_plugin/plaidml_op_reshape.cpp",
        "src/plaidml_plugin/plaidml_op_scaleshift.cpp",
        "src/plaidml_plugin/plaidml_op_sigmoid.cpp",
        "src/plaidml_plugin/plaidml_op_softmax.cpp",
        "src/plaidml_plugin/plaidml_plugin.cpp",
        "src/plaidml_plugin/plaidml_state.cpp",
        "src/plaidml_plugin/plaidml_util.cpp",
    ],
    hdrs = [
        "include/plaidml/plaidml_config.hpp",
    ],
    copts = ["-w"],
    includes = [
        "src/plaidml_plugin/any.hpp",
        "src/plaidml_plugin/plaidml_executable_network.hpp",
        "src/plaidml_plugin/plaidml_infer_request.hpp",
        "src/plaidml_plugin/plaidml_op.hpp",
        "src/plaidml_plugin/plaidml_op_clamp.hpp",
        "src/plaidml_plugin/plaidml_op_concat.hpp",
        "src/plaidml_plugin/plaidml_op_convolution.hpp",
        "src/plaidml_plugin/plaidml_op_crop.hpp",
        "src/plaidml_plugin/plaidml_op_eltwise.hpp",
        "src/plaidml_plugin/plaidml_op_fc.hpp",
        "src/plaidml_plugin/plaidml_op_norm.hpp",
        "src/plaidml_plugin/plaidml_op_permute.hpp",
        "src/plaidml_plugin/plaidml_op_pooling.hpp",
        "src/plaidml_plugin/plaidml_op_power.hpp",
        "src/plaidml_plugin/plaidml_op_relu.hpp",
        "src/plaidml_plugin/plaidml_op_reshape.hpp",
        "src/plaidml_plugin/plaidml_op_scaleshift.hpp",
        "src/plaidml_plugin/plaidml_op_sigmoid.hpp",
        "src/plaidml_plugin/plaidml_op_softmax.hpp",
        "src/plaidml_plugin/plaidml_plugin.hpp",
        "src/plaidml_plugin/plaidml_state.hpp",
        "src/plaidml_plugin/plaidml_util.hpp",
    ],
    deps = [
        "//plaidml/core",
        "//plaidml/edsl",
        "//plaidml/exec",
        "//plaidml/op",
        "@openvino//:inference_engine",
    ],
    alwayslink = 1,
)
