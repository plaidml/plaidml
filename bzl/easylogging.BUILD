package(default_visibility = ["@//visibility:public"])

exports_files(["LICENSE"])

cc_library(
    name = "easylogging",
    srcs = ["src/easylogging++.cc"],
    hdrs = ["src/easylogging++.h"],
    copts = [
        "-DELPP_THREAD_SAFE",
        "-DELPP_CUSTOM_COUT=std::cerr",
        "-DELPP_STL_LOGGING",
        "-DELPP_LOG_STD_ARRAY",
        "-DELPP_LOG_UNORDERED_MAP",
        "-DELPP_LOG_UNORDERED_SET",
        "-DELPP_NO_LOG_TO_FILE",
        "-DELPP_DISABLE_DEFAULT_CRASH_HANDLING",
        "-DELPP_WINSOCK2",
    ],
    includes = ["src"],
)
