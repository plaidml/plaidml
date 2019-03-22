package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license

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
        "@toolchain//:windows_x86_64": [],
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
    linkopts = select({
        "@toolchain//:windows_x86_64": ["-DEFAULTLIB:advapi32.lib"],
        "//conditions:default": [],
    }),
    deps = ["@zlib"],
)

genrule(
    name = "license",
    srcs = ["LICENSE"],
    outs = ["minizip-LICENSE"],
    cmd = "cp $(SRCS) $@",
)
