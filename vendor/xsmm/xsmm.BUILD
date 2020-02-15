package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE.md"])

cc_library(
    name = "xsmm",
    srcs = ["include/libxsmm_source.h"],
    hdrs = glob([
        "src/**/*.c",
        "src/**/*.h",
        "include/*",
    ]),
    defines = ["LIBXSMM_NO_BLAS"],
    includes = ["include"],
)
