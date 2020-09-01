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

py_binary(
    name = "conv_op_test_gen",
    srcs = ["conv_op_test_gen.py"],
    testonly = 1,
)

genrule(
    name = "conv_op_test_genrule",
    srcs = [],
    outs = ["conv_op_test.h.inc"],
    cmd = "XLA_FLAGS=\"--xla_dump_to=/tmp/hlo_module\" TF_XLA_FLAGS=\"--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=0\" $(location :conv_op_test_gen) --output $@",
    tools = [":conv_op_test_gen"],
    testonly = 1,
)

tf_cc_test(
    name = "conv_op_test",
    srcs = ["conv_op_test.cc", "conv_op_test.h.inc"],
    extra_copts = ["-fexceptions"],
    data = [":conv_op_test_genrule"],
    deps = [
        ":codegen_test",
	":filecheck",
	"//plaidml/bridge/tensorflow/service:compiler",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/tests:test_utils",
        "@org_tensorflow//tensorflow/compiler/xla/tests:hlo_test_base",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "dot_op_test",
    srcs = ["dot_op_test.cc"],
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

tf_cc_test(
    name = "logical_op_test",
    srcs = ["logical_op_test.cc"],
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

py_binary(
    name = "shape_op_test_gen",
    srcs = ["shape_op_test_gen.py"],
    testonly = 1,
)

genrule(
    name = "shape_op_test_genrule",
    srcs = [],
    outs = ["shape_op_test.h.inc"],
    cmd = "XLA_FLAGS=\"--xla_dump_to=/tmp/hlo_module\" TF_XLA_FLAGS=\"--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=0\" $(location :shape_op_test_gen) --output $@",
    tools = [":shape_op_test_gen"],
    testonly = 1,
)

tf_cc_test(
    name = "shape_op_test",
    srcs = ["shape_op_test.cc", "shape_op_test.h.inc"],
    extra_copts = ["-fexceptions"],
    data = [":shape_op_test_genrule"],
    deps = [
        ":codegen_test",
        ":filecheck",
        "//plaidml/bridge/tensorflow/service:compiler",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/tests:test_utils",
        "@org_tensorflow//tensorflow/compiler/xla/tests:hlo_test_base",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)
