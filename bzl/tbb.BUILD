package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

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
