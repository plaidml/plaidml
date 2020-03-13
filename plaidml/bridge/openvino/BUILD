package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license

cc_test(
    name = "smoke",
    tags = [
        "skip_windows",
    ],
    deps = [
        "//pmlc/testing:gtest_main",
        "@openvino//:smoke_tests",
    ],
)
