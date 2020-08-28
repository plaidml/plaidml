package(default_visibility = ["@//visibility:public"])

load("@org_tensorflow//tensorflow:tensorflow.bzl", "cc_header_only_library")

exports_files(
    [
        "LICENSE",
        "ACKNOWLEDGEMENTS",
        "configure",
        "configure.py",
    ],
)

cc_library(
    name = "xla_test",
    testonly = 1,
    hdrs = glob([
        "tensorflow/compiler/xla/test.h",
        "tensorflow/compiler/xla/tests/*.h"
    ]),
    includes = ["./tensorflow/compiler/xla"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)
