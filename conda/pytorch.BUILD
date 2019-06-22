package(default_visibility = ["//visibility:public"])

exports_files(["env"])

cc_library(
    name = "python_headers",
    hdrs = select({
        "@toolchain//:windows_x86_64": glob(["env/include/**/*.h"]),
        "//conditions:default": glob(["env/include/python3.7m/**/*.h"]),
    }),
    includes = select({
        "@toolchain//:windows_x86_64": ["env/include"],
        "//conditions:default": ["env/include/python3.7m"],
    }),
)

cc_library(
    name = "pytorch",
    srcs = select({
        "@toolchain//:windows_x86_64": [
            "env/Lib/python3.7/site-packages/torch/lib/c10.dll",
            "env/Lib/python3.7/site-packages/torch/lib/torch.dll",
        ],
        "@toolchain//:macos_x86_64": [
            "env/lib/python3.7/site-packages/torch/lib/libc10.dylib",
            "env/lib/python3.7/site-packages/torch/lib/libtorch.dylib",
        ],
        "//conditions:default": [
            "env/lib/python3.7/site-packages/torch/lib/libc10.so",
            "env/lib/python3.7/site-packages/torch/lib/libgomp-8bba0e50.so.1",
            "env/lib/python3.7/site-packages/torch/lib/libtorch.so",
        ],
    }),
    includes = [
        "env/lib/python3.7/site-packages/torch/include",
    ],
    deps = [
        ":python_headers",
        "@pybind11",
    ],
)
