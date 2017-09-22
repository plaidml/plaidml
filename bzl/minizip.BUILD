package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license

config_setting(
    name = "x64_windows",
    values = {"cpu": "x64_windows"},
)

SRCS = [
  "crypt.c",
  "ioapi.c",
  "ioapi_buf.c",
  "ioapi_mem.c",
  "minishared.c",
  "unzip.c",
  "zip.c",
]

HDRS = [
  "crypt.h",
  "ioapi.h",
  "ioapi_buf.h",
  "ioapi_mem.h",
  "minishared.h",
  "unzip.h",
  "zip.h",
]

cc_library(
    name = "minizip",
    srcs = SRCS,
    hdrs = HDRS,
    includes = [""],
    copts = select({
      ":x64_windows": [],
      "//conditions:default": [ 
        "-Wno-implicit-function-declaration", 
        "-Wno-int-conversion",
        "-Wno-format",
        "-D__USE_FILE_OFFSET64",
        "-D__USE_LARGEFILE64",
        "-D_LARGEFILE64_SOURCE",
        "-D_FILE_OFFSET_BIT=64",
      ],
    }),
    deps = ["//external:zlib"])
