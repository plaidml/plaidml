package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda_headers",
    hdrs = [
        %{cuda_headers}
    ],
    includes = [
        ".",
        "cuda/include",
        "cuda/include/crt",
    ],
)

cc_library(
    name = "cuda_driver",
    srcs = ["cuda/lib/%{cuda_driver_lib}"],
    includes = [
        ".",
        "cuda/include",
    ],
)

cc_library(
    name = "nvrtc",
    srcs = [
        "cuda/lib/%{nvrtc_lib}",
        "cuda/lib/%{nvrtc_builtins_lib}",
    ],
    includes = [
        ".",
        "cuda/include",
    ],
    linkstatic = 1,
)

%{cuda_include_genrules}
