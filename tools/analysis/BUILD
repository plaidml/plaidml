package(default_visibility = ["//visibility:public"])

load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("//bzl:plaidml.bzl", "plaidml_py_library")

py_binary(
    name = "analysis",
    srcs = [
        "__init__.py",
        "main.py",
    ],
    data = ["Analysis.ipynb"],
    main = "main.py",
    deps = [
        ":util",
        "//tile/hal/opencl:proto_py",
        "//tile/platform/local_machine:proto_py",
        "//tile/proto:hal_py",
        "//tile/proto:schedule_py",
    ],
)

py_test(
    name = "analysis_test",
    srcs = ["analysis_test.py"],
    data = [
        "Analysis.ipynb",
        "testdata/small_mobilenet_coco_log.gz",
    ],
    tags = ["jupyter"],
    deps = [":analysis"],
)

py_binary(
    name = "kernel_times",
    srcs = ["kernel_times.py"],
    deps = [":analysis"],
)

plaidml_py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//base/context:proto_py",
        "//base/eventing/file:proto_py",
    ],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    data = ["testdata/eventlog.gz"],
    tags = ["jupyter"],
    deps = [":util"],
)
