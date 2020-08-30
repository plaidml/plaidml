load("@rules_cc//cc:defs.bzl", "cc_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "codegen_test",
    testonly = True,
    srcs = ["codegen_test.cc"],
    hdrs = ["codegen_test.h"],
    deps = [
        "//plaidml:testenv",
        "//plaidml/bridge/tensorflow/service:compiler",
        "@org_tensorflow//tensorflow/compiler/xla/tests:hlo_test_base",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "filecheck",
    testonly = True,
    srcs = ["filecheck.cc"],
    hdrs = ["filecheck.h"],
    data = [
        "@llvm-project//llvm:FileCheck",
    ],
    deps = [
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        "@org_tensorflow//tensorflow/compiler/xla:types",
        "@org_tensorflow//tensorflow/compiler/xla:util",
        "@org_tensorflow//tensorflow/core:lib",
	"@bazel_tools//tools/cpp/runfiles",
    ],
)

tf_cc_test(
    name = "compare_op_test",
    srcs = ["compare_op_test.cc"],
    extra_copts = ["-fexceptions"],
    deps = [
        ":codegen_test",
        ":filecheck",
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

tf_cc_test(
    name = "eltwise_op_test",
    srcs = ["eltwise_op_test.cc"],
    extra_copts = ["-fexceptions"],
    deps = [
        ":codegen_test",
	":filecheck",
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
