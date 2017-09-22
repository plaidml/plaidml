config_setting(
    name = "x64_windows",
    values = {"cpu": "x64_windows"},
)

cc_library(
    name = "lib",
    srcs = [
        "icd.c",
        "icd_dispatch.c",
    ] + select({
        "//:x64_windows": ["icd_windows.c"],
        "//conditions:default": ["icd_linux.c"],
    }),
    hdrs = [
        "icd.h",
        "icd_dispatch.h",
    ],
    linkopts = select({
        ":x64_windows": [],
        "//conditions:default": ["-pthread", "-ldl"],
    }),
    deps = [
        "//external:opencl_headers",
    ],
    visibility = ["//visibility:public"],
    nocopts = select({
        "//:x64_windows": "/DNOGDI",
        "//conditions:default": "",
    })
)
