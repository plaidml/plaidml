cc_library(
    name = "lib",
    srcs = [
        "icd.c",
        "icd_dispatch.c",
    ] + select({
        "@toolchain//:windows_x86_64": ["icd_windows.c"],
        "//conditions:default": ["icd_linux.c"],
    }),
    hdrs = [
        "icd.h",
        "icd_dispatch.h",
    ],
    linkopts = select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": [
            "-pthread",
            "-ldl",
        ],
    }),
    nocopts = select({
        "@toolchain//:windows_x86_64": "/DNOGDI",
        "//conditions:default": "",
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@opencl_headers_repo//:inc",
        "@opengl_repo//:inc",
    ],
)
