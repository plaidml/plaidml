package(default_visibility = ["//visibility:public"])

cc_library(
    name = "VulkanRuntime",
    srcs = [
        "VulkanRuntime.cpp",
        "vulkan-runtime-wrappers.cpp",
    ],
    hdrs = [
        "VulkanRuntime.h",
    ],
    deps = [
        "//mlir:Pass",
        "//mlir:SPIRVLowering",
        "@vulkan_headers//:vulkan_headers",
    ],
    alwayslink = 1,
)
