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
        "@com_intel_plaidml//toolchain:macos_x86_64": [],
        "@com_intel_plaidml//toolchain:windows_x86_64": [],
        "//conditions:default": ["-pthread"],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gtest_main",
    srcs = ["googlemock/src/gmock_main.cc"],
    linkopts = select({
        "@com_intel_plaidml//toolchain:macos_x86_64": [],
        "@com_intel_plaidml//toolchain:windows_x86_64": [],
        "//conditions:default": ["-pthread"],
    }),
    visibility = ["//visibility:public"],
    deps = [":gtest"],
)

genrule(
    name = "license",
    srcs = ["googlemock/LICENSE"],
    outs = ["gmock-LICENSE"],
    cmd = "cp $(SRCS) $@",
)
