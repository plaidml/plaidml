load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["@//visibility:public"])

exports_files(["LICENSE.txt"])

cc_library(
    name = "half",
    hdrs = ["include/half.hpp"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)
