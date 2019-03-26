# Description:
#   The Boost library collection (http://www.boost.org)
#
# Most Boost libraries are header-only, in which case you only need to depend
# on :boost. If you need one of the libraries that has a separately-compiled
# implementation, depend on the appropriate libs rule.

package(default_visibility = ["@//visibility:public"])

licenses(["notice"])  # Boost software license

cc_library(
    name = "boost",
    hdrs = glob([
        "boost/**/*.hpp",
        "boost/**/*.h",
        "boost/**/*.ipp",
    ]),
    defines = [
        "BOOST_ERROR_CODE_HEADER_ONLY",
        "BOOST_SYSTEM_NO_DEPRECATED",
        "BOOST_THREAD_BUILD_LIB",
        "BOOST_THREAD_VERSION=4",
        "BOOST_THREAD_PROVIDES_EXECUTORS",
        "BOOST_ALL_NO_LIB",
    ] + select({
        "@toolchain//:macos_x86_64": ["BOOST_ASIO_HAS_STD_STRING_VIEW"],
        "//conditions:default": [],
    }),
    includes = ["."],
)

cc_library(
    name = "filesystem",
    srcs = glob(["libs/filesystem/src/*.cpp"]),
    deps = [
        ":boost",
        ":system",
    ],
)

cc_library(
    name = "stacktrace",
    defines = select({
        "@toolchain//:macos_x86_64": ["BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@toolchain//:macos_x86_64": [],
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": ["-ldl"],
    }),
)

cc_library(
    name = "program_options",
    srcs = glob(["libs/program_options/src/*.cpp"]),
    deps = [
        ":boost",
    ],
)

cc_library(
    name = "regex",
    srcs = glob([
        "libs/regex/src/*.cpp",
        "libs/regex/src/*.hpp",
    ]),
    deps = [
        ":boost",
    ],
)

cc_library(
    name = "system",
    srcs = glob(["libs/system/src/*.cpp"]),
    deps = [
        ":boost",
    ],
)

cc_library(
    name = "thread",
    srcs = glob(["libs/thread/src/*.cpp"]) +
           select({
               "@toolchain//:windows_x86_64": glob(["libs/thread/src/win32/*.cpp"]),
               "//conditions:default": glob(
                   ["libs/thread/src/pthread/*.cpp"],
                   exclude = ["libs/thread/src/pthread/once.cpp"],
               ),
           }),
    linkopts = select({
        "@toolchain//:macos_x86_64": [],
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": ["-pthread"],
    }),
    deps = [
        ":boost",
        ":system",
    ],
)

genrule(
    name = "license",
    srcs = ["LICENSE_1_0.txt"],
    outs = ["boost-LICENSE"],
    cmd = "cp $(SRCS) $@",
)
