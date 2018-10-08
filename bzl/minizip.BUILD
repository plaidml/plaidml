package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license

config_setting(
    name = "x64_windows",
    values = {"cpu": "x64_windows"},
)

cc_library(
    name = "minizip",
    srcs = [
        "crypt.c",
        "ioapi.c",
        "ioapi_buf.c",
        "ioapi_mem.c",
        "minishared.c",
        "unzip.c",
        "zip.c",
    ],
    hdrs = [
        "crypt.h",
        "ioapi.h",
        "ioapi_buf.h",
        "ioapi_mem.h",
        "minishared.h",
        "unzip.h",
        "zip.h",
    ],
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
    includes = ["."],
    deps = ["@zlib_archive//:zlib"],
)
