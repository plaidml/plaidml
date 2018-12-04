package(default_visibility = ["@//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_library(
    name = "half",
    hdrs = ["include/half.hpp"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "sdk_includes",
    srcs = ["include/half.hpp"],
    package_dir = "include",
    visibility = ["//visibility:public"],
)

exports_files(["LICENSE.txt"])
