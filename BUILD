# Copyright 2019 Intel Corporation
#
# For build instructions, see <docs/building.md>.

package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "pkg",
    srcs = [
        "//plaidbench:wheel",
        "//plaidml:wheel",
        "//plaidml/keras:wheel",
    ],
    extension = "tar.gz",
)
