load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

exports_files(["env"])

filegroup(
    name = "perl",
    srcs = ["env/bin/perl"],
)

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

cc_library(
    name = "opencv_headers",
    hdrs = glob(["env/include/opencv4/**/*.h"]),
    includes = ["env/include/opencv4"],
)

cc_library(
    name = "opencv_core",
    srcs = select({
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": [
            "env/lib/libopencv_core.so",
            "env/lib/libopencv_core.so.4.2",
            "env/lib/liblapack.so.3",
            "env/lib/libcblas.so.3",
            "env/lib/libgomp.so.1",
            "env/lib/libgfortran.so.4",
            "env/lib/libquadmath.so.0",
            "env/lib/libopenblas.so.0",
        ],
    }),
    deps = [":opencv_headers"],
)

cc_library(
    name = "opencv_imgproc",
    srcs = select({
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": [
            "env/lib/libopencv_imgproc.so",
            "env/lib/libopencv_imgproc.so.4.2",
        ],
    }),
    deps = [":opencv_core"],
)

cc_library(
    name = "opencv_imgcodecs",
    srcs = select({
        "@bazel_tools//src/conditions:darwin_x86_64": [],
        "//conditions:default": [
            "env/lib/libopencv_imgcodecs.so",
            "env/lib/libopencv_imgcodecs.so.4.2",
            "env/lib/libjpeg.so.9",
            "env/lib/libwebp.so.7",
            "env/lib/libtiff.so.5",
            "env/lib/libjasper.so.1",
            "env/lib/libz.so.1",
            "env/lib/libpng16.so.16",
            "env/lib/libzstd.so.1",
            "env/lib/liblzma.so.5",
        ],
    }),
    deps = [":opencv_imgproc"],
)
