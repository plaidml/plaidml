includes = ["."]

package(default_visibility = ["@//visibility:public"])

exports_files(["LICENCE.txt"])

cc_library(
    name = "easylogging",
    srcs = ["easylogging++.cc"],
    hdrs = ["easylogging++.h"],
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
    includes = ["."],
    visibility = ["//visibility:public"],
)

genrule(
    name = "license",
    srcs = ["LICENCE.txt"],
    outs = ["easylogging-LICENSE"],
    cmd = "cp $(SRCS) $@",
)
