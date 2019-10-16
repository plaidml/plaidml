package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license

exports_files(["documentation/LICENSE.md"])

cc_library(
    name = "tbb",
    srcs = glob([
        "tbb/**/*.cpp",
        "tbb/**/*.cc",
        "tbbmalloc/**/*.cpp",
        "tbbmalloc/**/*.cc",
    ]),
    hdrs = glob([
        "include/serial/**",
        "include/tbb/**/**",
    ]),
    includes = ["include"],
)

genrule(
    name = "license",
    srcs = ["doc/copyright_brand_disclaimer_doxygen.txt"],
    outs = ["tbb-LICENSE"],
    cmd = "cp $(SRCS) $@",
)
