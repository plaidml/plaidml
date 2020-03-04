package(default_visibility = ["//visibility:public"])

cc_library(
    name = "marl",
    srcs = [],
)

cc_library(
    name = "vk_swiftshader",
    srcs = glob([
        "src/Vulkan/*.cpp",
        "src/Vulkan/*.h",
        "src/Vulkan/*.hpp",
        "src/System/Build.cpp",
        "src/System/Build.hpp",
        "src/System/CPUID.cpp",
        "src/System/CPUID.hpp",
        "src/System/Configurator.cpp",
        "src/System/Configurator.hpp",
        "src/System/Debug.cpp",
        "src/System/Debug.hpp",
        "src/System/Half.cpp",
        "src/System/Half.hpp",
        "src/System/Math.cpp",
        "src/System/Math.hpp",
        "src/System/Memory.cpp",
        "src/System/Memory.hpp",
        "src/System/Socket.cpp",
        "src/System/Socket.hpp",
        "src/System/Synchronization.hpp",
        "src/System/Timer.cpp",
        "src/System/Timer.hpp",
        "src/Device/*.cpp",
        "src/Device/*.hpp",
        "src/Pipeline/*.cpp",
        "src/Pipeline/*.hpp",
        "src/WSI/VkSurfaceKHR.cpp",
        "src/WSI/VkSurfaceKHR.hpp",
        "src/WSI/VkSwapchainKHR.cpp",
        "src/WSI/VkSwapchainKHR.hpp",
    ]) + select({
        "@bazel_tools//src/conditions:darwin_x86_64": [
            "src/WSI/MetalSurface.mm",
            "src/WSI/MetalSurface.h",
        ],
        "@bazel_tools//src/conditions:windows": [
            "src/Vulkan/Vulkan.rc",
            "src/WSI/Win32SurfaceKHR.cpp",
            "src/WSI/Win32SurfaceKHR.hpp",
        ],
        "//conditions:default": [
            "src/System/Linux/MemFd.cpp",
            "src/System/Linux/MemFd.hpp",
        ],
    }),
    hdrs = glob([
        "include/vulkan/*.h",
    ]),
    defines = select({
        "@bazel_tools//src/conditions:darwin_x86_64": ["VK_USE_PLATFORM_MACOS_MVK"],
        "@bazel_tools//src/conditions:windows": ["VK_USE_PLATFORM_WIN32_KHR"],
        "//conditions:default": [],
    }),
    includes = [
        "include",
        "src",
    ],
)
