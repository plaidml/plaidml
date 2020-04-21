package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE.md"])

cc_library(
    name = "xsmm",
    srcs = glob(
        [
            "src/*.c",
            "src/*.h",
        ],
        exclude = ["src/libxsmm_generator_gemm_driver.c"],
    ),
    hdrs = glob([
        "include/*",
        "src/template/*.tpl.c",
    ]),
    defines = [
        "__BLAS=0",
        "LIBXSMM_DEFAULT_CONFIG",
    ],
    includes = ["include"],
)
