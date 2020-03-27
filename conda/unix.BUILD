package(default_visibility = ["//visibility:public"])

exports_files(["env"])

filegroup(
    name = "python",
    srcs = ["env/bin/python"],
)

cc_library(
    name = "python_headers",
    hdrs = glob(["env/include/python3.7m/**/*.h"]),
    includes = ["env/include/python3.7m"],
)

cc_library(
    name = "pytorch",
    srcs = select({
        "@bazel_tools//src/conditions:darwin_x86_64": [
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
