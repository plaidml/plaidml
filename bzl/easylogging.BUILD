config_setting(
    name = "x64_windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "easylogging",
    srcs = ["src/easylogging++.cc"],
    hdrs = ["src/easylogging++.h"],
    copts = [
        "-std=c++11",
        "-DELPP_THREAD_SAFE",
        "-DELPP_CUSTOM_COUT=std::cerr",
        "-DELPP_STL_LOGGING",
        "-DELPP_LOG_STD_ARRAY",
        "-DELPP_LOG_UNORDERED_MAP",
        "-DELPP_LOG_UNORDERED_SET",
        "-DELPP_NO_LOG_TO_FILE",
        "-DELPP_DISABLE_DEFAULT_CRASH_HANDLING",
        "-DELPP_WINSOCK2",
    ] + select({
        ":x64_windows": [],
        "//conditions:default": ["-DELPP_FEATURE_CRASH_LOG"],
    }),
    includes = ["src"],
    visibility = ["//visibility:public"],
)
