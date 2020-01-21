package(default_visibility = ["@//visibility:public"])

exports_files(["googlemock/LICENSE"])

cc_library(
    name = "gtest",
    srcs = [
        "googlemock/src/gmock-all.cc",
        "googletest/src/gtest-all.cc",
    ],
    hdrs = glob([
        "**/*.h",
        "googletest/src/*.cc",
        "googlemock/src/*.cc",
    ]),
    includes = [
        "googlemock",
        "googlemock/include",
        "googletest",
        "googletest/include",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": ["-pthread"],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gtest_main",
    srcs = ["googlemock/src/gmock_main.cc"],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": ["-pthread"],
    }),
    visibility = ["//visibility:public"],
    deps = [":gtest"],
)
