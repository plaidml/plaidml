load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "lz_wrapper",
    srcs = [
            "utils.cc",
    ],
    includes = ["include",],
    hdrs = [
            "utils.h",
    ],
    copts = [
      "-std=c++11",
    #  "-I/usr/local/include",
    ],
    linkopts = [
      "-lze_loader",
      "-L/usr/local/lib",
    ],
    #deps = [
    #  "@level_zero_local//:headers",
    #],
    visibility = ["//visibility:public"],
    #linkstatic = 1,
)
