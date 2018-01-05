# Description:
#   The Boost library collection (http://www.boost.org)
#
# Most Boost libraries are header-only, in which case you only need to depend
# on :boost. If you need one of the libraries that has a separately-compiled
# implementation, depend on the appropriate libs rule.

package(default_visibility = ["@//visibility:public"])

licenses(["notice"])  # Boost software license

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "x64win",
    values = {"cpu": "x64win"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "x64_windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

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
        "BOOST_ALL_NO_LIB",
    ],
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
    name = "iostreams",
    srcs = glob(["libs/iostreams/src/*.cpp"]),
    deps = [
        ":boost",
        "@bzip2_archive//:bz2lib",
        "@zlib_archive//:zlib",
    ],
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
               "//:x64win": glob(["libs/thread/src/win32/*.cpp"]),
               "//:x64_windows": glob(["libs/thread/src/win32/*.cpp"]),
               "//conditions:default": glob(
                   ["libs/thread/src/pthread/*.cpp"],
                   exclude = ["libs/thread/src/pthread/once.cpp"],
               ),
           }),
    linkopts = select({
        "//:darwin": [],
        "//:x64win": [],
        "//:x64_windows": [],
        "//conditions:default": ["-pthread"],
    }),
    deps = [
        ":boost",
        ":system",
    ],
)
