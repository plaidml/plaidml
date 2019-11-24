# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

cc_library(
    name = "tbb",
    srcs = glob([
        "src/rml/client/rml_tbb.cpp",
        "src/rml/**/*.h",
        "src/tbb/*.cpp",
        "src/tbb/*.h",
    ]) + select({
        "@com_intel_plaidml//toolchain:windows_x86_64": [
            "@com_intel_plaidml//vendor/tbb:gen_cpu_ctl_env.cc",
        ],
        "//conditions:default": [],
    }),
    hdrs = glob([
        "include/serial/**",
        "include/tbb/**/**",
        "build/vs2013/version_string.ver",
    ]),
    copts = [
        "-Iexternal/tbb/build/vs2013",
        "-Iexternal/tbb/src",
    ],
    defines = [
        "__TBB_DYNAMIC_LOAD_ENABLED=0",
        "__TBB_SOURCE_DIRECTLY_INCLUDED=1",
    ] + select({
        "@com_intel_plaidml//toolchain:windows_x86_64": [
            "__TBB_CPU_CTL_ENV_PRESENT=1",
            "__TBB_x86_64=1",
            "TBB_USE_THREADING_TOOLS",
            "USE_WINTHREAD",
        ],
        "//conditions:default": [
            "__TBB_BUILD=1",
            "TBB_SUPPRESS_DEPRECATED_MESSAGES=1",
            "USE_PTHREAD",
        ],
    }),
    includes = ["include"],
    alwayslink = 1,
)
