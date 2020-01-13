# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("@bazel_skylib//rules:copy_file.bzl", "copy_file")

exports_files(["LICENSE"])

# NOTE: we must copy this file because bazel cannot handle directories with 'build' in them
copy_file(
    name = "version_string",
    src = "build/vs2013/version_string.ver",
    out = "version_string.ver",
)

copy_file(
    name = "gen_cpu_ctl_env",
    src = "@com_intel_plaidml//vendor/tbb:gen_cpu_ctl_env.cc",
    out = "gen_cpu_ctl_env.cc",
)

cc_library(
    name = "tbb",
    srcs = glob([
        "src/rml/client/rml_tbb.cpp",
        "src/rml/**/*.h",
        "src/tbb/*.cpp",
        "src/tbb/*.h",
    ]) + select({
        "@bazel_tools//src/conditions:windows": [
            ":gen_cpu_ctl_env",
        ],
        "//conditions:default": [],
    }),
    hdrs = glob([
        "include/serial/**",
        "include/tbb/**/**",
    ]) + [
        ":version_string",
    ],
    copts = [
        "-Iexternal/tbb/src",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin_x86_64": [
            "-mrtm",
        ],
        "//conditions:default": [
            "-mrtm",
            # this prevents segfaults
            # see: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
            "-flifetime-dse=1",
        ],
    }),
    defines = [
        "TBB_SUPPRESS_DEPRECATED_MESSAGES=1",
        "__TBB_BUILD=1",
    ],
    includes = ["include"],
    linkstatic = 1,
    local_defines = [
        "__TBB_DYNAMIC_LOAD_ENABLED=0",
        "__TBB_SOURCE_DIRECTLY_INCLUDED=1",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "__TBB_CPU_CTL_ENV_PRESENT=1",
            "__TBB_x86_64=1",
            "TBB_USE_THREADING_TOOLS",
            "USE_WINTHREAD",
        ],
        "//conditions:default": [
            "USE_PTHREAD",
        ],
    }),
    alwayslink = 1,
)
