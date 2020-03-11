package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license

cc_test(
    name = "smoke",
    deps = [
      "//pmlc/testing:gtest_main",
      "@openvino//:smoke_tests"
    ],
    tags = [
      "skip_windows"
    ]
)