package(default_visibility = ["@//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_library(
    name = "half",
    hdrs = ["include/half.hpp"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "license",
    srcs = ["LICENSE.txt"],
    outs = ["half-LICENSE"],
    cmd = "cp $(SRCS) $@",
)
