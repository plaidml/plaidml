
# Description:
#   The Boost library collection (http://www.boost.org)
#
# Most Boost libraries are header-only, in which case you only need to depend
# on :boost. If you need one of the libraries that has a separately-compiled
# implementation, depend on the appropriate libs rule.

package(default_visibility = ["@//visibility:public"])

licenses(["notice"])  # Boost software license

prefix_dir = "boost_1_63_0"

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
    prefix_dir + "/boost/**/*.hpp",
    prefix_dir + "/boost/**/*.h",
    prefix_dir + "/boost/**/*.ipp",
    ]),
  includes = [prefix_dir],
  defines = [
    "BOOST_ERROR_CODE_HEADER_ONLY",
    "BOOST_SYSTEM_NO_DEPRECATED",
    "BOOST_THREAD_BUILD_LIB",
    "BOOST_THREAD_VERSION=4",
    "BOOST_ALL_NO_LIB",
  ],
)

cc_library(
  name = "filesystem",
  srcs = glob([prefix_dir + "/libs/filesystem/src/*.cpp"]),
  deps = [
    ":boost",
    ":system",
  ],
)

cc_library(
  name = "iostreams",
  srcs = glob([prefix_dir + "/libs/iostreams/src/*.cpp"]),
  deps = [
    ":boost",
    "@bzip2_archive//:bz2lib",
    "@zlib_archive//:zlib",
  ],
)

cc_library(
  name = "program_options",
  srcs = glob([prefix_dir + "/libs/program_options/src/*.cpp"]),
  deps = [
    ":boost",
  ],
)

cc_library(
  name = "regex",
  srcs = glob([prefix_dir + "/libs/regex/src/*.cpp", prefix_dir + "/libs/regex/src/*.hpp"]),
  deps = [
    ":boost",
  ],
)

cc_library(
  name = "system",
  srcs = glob([prefix_dir + "/libs/system/src/*.cpp"]),
  deps = [
    ":boost",
  ],
)

cc_library(
  name = "thread",
  srcs = glob([prefix_dir + "/libs/thread/src/*.cpp"]) +
         select({
           "//:x64win": glob([prefix_dir + "/libs/thread/src/win32/*.cpp"]),
           "//:x64_windows": glob([prefix_dir + "/libs/thread/src/win32/*.cpp"]),
           "//conditions:default":
              glob([prefix_dir + "/libs/thread/src/pthread/*.cpp"],
                   exclude=[prefix_dir + "/libs/thread/src/pthread/once.cpp"]),
         }),
  deps = [
    ":boost",
    ":system",
  ],
  linkopts = select({
    "//:darwin": [],
    "//:x64win": [],
    "//:x64_windows": [],
    "//conditions:default": ["-pthread"],
  }),
)
