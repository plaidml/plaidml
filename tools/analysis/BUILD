package(default_visibility = ["//visibility:public"])

load("//bzl:conda.bzl", "conda_binary", "conda_test")
load("//bzl:plaidml.bzl", "plaidml_py_library")
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

conda_binary(
    name = "analysis",
    srcs = [
        "__init__.py",
        "main.py",
    ],
    data = ["Analysis.ipynb"],
    env = "@com_intel_plaidml_conda_analysis//:env",
    main = "main.py",
    deps = [
        ":util",
        "//tile/hal/opencl:proto_py",
        "//tile/platform/local_machine:proto_py",
        "//tile/proto:hal_py",
        "//tile/proto:schedule_py",
    ],
)

conda_test(
    name = "analysis_test",
    srcs = ["analysis_test.py"],
    data = [
        "Analysis.ipynb",
        "testdata/small_mobilenet_coco_log.gz",
    ],
    env = "@com_intel_plaidml_conda_analysis//:env",
    tags = ["jupyter"],
    deps = [":analysis"],
)

conda_binary(
    name = "kernel_times",
    srcs = ["kernel_times.py"],
    env = "@com_intel_plaidml_conda_analysis//:env",
    deps = [":analysis"],
)

pkg_tar(
    name = "pkg",
    srcs = glob(["**/*"]),
    package_dir = "tools/analysis",
    strip_prefix = ".",
)

plaidml_py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//base/context:proto_py",
        "//base/eventing/file:proto_py",
    ],
)

conda_test(
    name = "util_test",
    srcs = ["util_test.py"],
    data = ["testdata/eventlog.gz"],
    env = "@com_intel_plaidml_conda_analysis//:env",
    tags = ["jupyter"],
    deps = [":util"],
)
