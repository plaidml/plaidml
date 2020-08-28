load("@rules_cc//cc:defs.bzl", "cc_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "plaidml_codegen_test",
    testonly = True,
    srcs = ["plaidml_codegen_test.cc"],
    hdrs = ["plaidml_codegen_test.h"],
    deps = [
        "//plaidml:testenv",
        "//plaidml/bridge/tensorflow/service:compiler",
        "@org_tensorflow//tensorflow/compiler/xla/service:llvm_compiler",
        "@org_tensorflow//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "@org_tensorflow//tensorflow/compiler/xla/tests:filecheck",
        "@org_tensorflow//tensorflow/compiler/xla/tests:hlo_test_base",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "plaidml_eltwise_op_test",
    srcs = ["plaidml_eltwise_op_test.cc"],
    extra_copts = ["-fexceptions"],
    deps = [
        ":plaidml_codegen_test",
        "//plaidml/bridge/tensorflow/service:compiler",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/tests:hlo_test_base",
        "@org_tensorflow//tensorflow/compiler/xla/tests:test_utils",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
    ],
)
