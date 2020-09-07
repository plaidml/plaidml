package(default_visibility = ["//visibility:public"])

cc_library(
    name = "volk",
    srcs = ["volk.c"],
    hdrs = ["volk.h"],
    defines = select({
        "@bazel_tools//src/conditions:darwin_x86_64": ["VK_USE_PLATFORM_MACOS_MVK"],
        "@bazel_tools//src/conditions:windows": ["VK_USE_PLATFORM_WIN32_KHR"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    deps = ["@vulkan_headers"],
)
