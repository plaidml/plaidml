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
    copts = ["-w"],
    defines = ["__BLAS=0"],
    includes = [
        "include",
        "src",
    ],
    alwayslink = 1,
)

genrule(
    name = "license",
    srcs = ["documentation/LICENSE.md"],
    outs = ["xsmm-LICENSE"],
    cmd = "cp $(SRCS) $@",
)
