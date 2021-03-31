package(default_visibility = ["//visibility:public"])

# This file contains cmake-generated definitions.  We use local defines on the cc_library rule instead.
genrule(
    name = "icd_cmake_config_gen",
    srcs = [],
    outs = ["loader/icd_cmake_config.h"],
    cmd = "echo '' > \"$@\"",
    cmd_ps = "echo '' > \"$@\"",
    visibility = ["//visibility:private"],
)

opencl_icd_loader_srcs = glob([
    "loader/*.c",
    "loader/*.h",
]) + [
    "loader/icd_cmake_config.h",
]

# Linux specific configuration
opencl_icd_loader_lnx_srcs = glob([
    "loader/linux/*.c",
])

opencl_icd_loader_lnx_lnks = [
    "-ldl",
    "-lpthread",
]

# Windows specific configuration
opencl_icd_loader_win_srcs = glob([
    "loader/windows/*.c",
    "loader/windows/*.cpp",
    "loader/windows/*.h",
])

opencl_icd_loader_win_lnks = [
    "cfgmgr32.lib",
    "runtimeobject.lib",
    "Advapi32.lib",
    "ole32.lib",
]

# Select configuration based on system
opencl_icd_loader_srcs += select({
    "@bazel_tools//src/conditions:windows": opencl_icd_loader_win_srcs,
    "//conditions:default": opencl_icd_loader_lnx_srcs,
})

opencl_icd_loader_lnks = select({
    "@bazel_tools//src/conditions:windows": opencl_icd_loader_win_lnks,
    "//conditions:default": opencl_icd_loader_lnx_lnks,
})

cc_library(
    name = "opencl_icd_loader",
    srcs = opencl_icd_loader_srcs,
    copts = ["-w"],
    defines = ["CL_TARGET_OPENCL_VERSION=220"],
    includes = ["loader"],
    linkopts = opencl_icd_loader_lnks,
    linkstatic = 1,
    local_defines = select({
        "@bazel_tools//src/conditions:linux_x86_64": ["HAVE_SECURE_GETENV"],
        "//conditions:default": [],
    }),
    deps = ["@opencl_headers"],
)
