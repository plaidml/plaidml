load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_copts")

cc_library(
    name = "plaidml_codegen_test",
    testonly = True,
    srcs = ["plaidml_codegen_test.cc"],
    hdrs = ["plaidml_codegen_test.h"],
    deps = [
        "//plaidml/bridge/tensorflow/service:compiler",
        "//plaidml:testenv",
        "@org_tensorflow//:xla_test",
        "@org_tensorflow//tensorflow/compiler/xla:xla_headers_lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "plaidml_eltwise_op_test",
    srcs = ["plaidml_eltwise_op_test.cc"],
    extra_copts = ["-fexceptions"],
    deps = [
        "//plaidml/bridge/tensorflow/service:compiler",
        ":plaidml_codegen_test",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

