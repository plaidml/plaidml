load("@bazel_skylib//lib:dicts.bzl", "dicts")
load("@bazel_skylib//rules:run_binary.bzl", "run_binary")
load("@com_intel_plaidml//vendor/llvm:llvm.bzl", "cmake_var_string", "expand_cmake_vars")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Define OpenMP-specific CMake vars
cmake_vars_base = {
    "LIBOMP_LIB_FILE": "libomp.so",
    "LIBOMP_VERSION_MAJOR": "5",
    "LIBOMP_VERSION_MINOR": "0",
    "LIBOMP_VERSION_BUILD": "20140926",
    "LIBOMP_BUILD_DATE": "20140926",
}

cmake_vars_windows = {
    "MSVC": 1,
}

cmake_vars = select({
    "@bazel_tools//src/conditions:darwin": cmake_var_string(cmake_vars_base),
    "@bazel_tools//src/conditions:darwin_x86_64": cmake_var_string(cmake_vars_base),
    "@bazel_tools//src/conditions:windows": cmake_var_string(
        dicts.add(
            cmake_vars_base,
            cmake_vars_windows,
        ),
    ),
    "//conditions:default": cmake_var_string(cmake_vars_base),
})

# Auto-generated files (generated using configure_file in CMake)
expand_cmake_vars(
    name = "omp_gen",
    src = "runtime/src/include/omp.h.var",
    cmake_vars = cmake_vars,
    dst = "runtime/src/omp.h",
)

expand_cmake_vars(
    name = "kmp_config_gen",
    src = "runtime/src/kmp_config.h.cmake",
    cmake_vars = cmake_vars,
    dst = "runtime/src/kmp_config.h",
)

expand_cmake_vars(
    name = "omp_tools_gen",
    src = "runtime/src/include/omp-tools.h.var",
    cmake_vars = cmake_vars,
    dst = "runtime/src/omp-tools.h",
)

run_binary(
    name = "kmp_i18n_id",
    srcs = [
        "runtime/src/i18n/en_US.txt",
        "runtime/tools/message-converter.pl",
    ],
    outs = ["runtime/src/kmp_i18n_id.inc"],
    args = [
        "$(location runtime/tools/message-converter.pl)",
    ] + select({
        "@bazel_tools//src/conditions:darwin": ["--os=mac"],
        "@bazel_tools//src/conditions:windows": ["--os=win"],
        "//conditions:default": ["--os=lin"],
    }) + [
        "--prefix=kmp_i18n",
        "--enum=$(location runtime/src/kmp_i18n_id.inc)",
        "$(location runtime/src/i18n/en_US.txt)",
    ],
    tool = "@com_intel_plaidml_conda//:perl",
)

run_binary(
    name = "kmp_i18n_default",
    srcs = [
        "runtime/src/i18n/en_US.txt",
        "runtime/tools/message-converter.pl",
    ],
    outs = ["runtime/src/kmp_i18n_default.inc"],
    args = [
        "$(location runtime/tools/message-converter.pl)",
    ] + select({
        "@bazel_tools//src/conditions:darwin": ["--os=mac"],
        "@bazel_tools//src/conditions:windows": ["--os=win"],
        "//conditions:default": ["--os=lin"],
    }) + [
        "--prefix=kmp_i18n",
        "--default=$(location runtime/src/kmp_i18n_default.inc)",
        "$(location runtime/src/i18n/en_US.txt)",
    ],
    tool = "@com_intel_plaidml_conda//:perl",
)

cc_library(
    name = "openmp",
    srcs = [
        "runtime/src/kmp_config.h",
        "runtime/src/kmp_i18n_id.inc",
        "runtime/src/kmp_i18n_default.inc",
        "runtime/src/omp.h",
        "runtime/src/omp-tools.h",
        "runtime/src/kmp_alloc.cpp",
        "runtime/src/kmp_atomic.cpp",
        "runtime/src/kmp_csupport.cpp",
        "runtime/src/kmp_debug.cpp",
        "runtime/src/kmp_itt.cpp",
        "runtime/src/kmp_environment.cpp",
        "runtime/src/kmp_error.cpp",
        "runtime/src/kmp_global.cpp",
        "runtime/src/kmp_i18n.cpp",
        "runtime/src/kmp_io.cpp",
        "runtime/src/kmp_runtime.cpp",
        "runtime/src/kmp_settings.cpp",
        "runtime/src/kmp_str.cpp",
        "runtime/src/kmp_tasking.cpp",
        "runtime/src/kmp_threadprivate.cpp",
        "runtime/src/kmp_utility.cpp",
        "runtime/src/kmp_barrier.cpp",
        "runtime/src/kmp_wait_release.cpp",
        "runtime/src/kmp_affinity.cpp",
        "runtime/src/kmp_dispatch.cpp",
        "runtime/src/kmp_lock.cpp",
        "runtime/src/kmp_sched.cpp",
        "runtime/src/kmp_taskdeps.cpp",
        "runtime/src/kmp_cancel.cpp",
        "runtime/src/kmp_ftn_cdecl.cpp",
        "runtime/src/kmp_ftn_extra.cpp",
        "runtime/src/kmp_version.cpp",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "runtime/src/z_Windows_NT_util.cpp",
            "runtime/src/z_Windows_NT-586_asm.asm",
            "runtime/src/z_Windows_NT-586_util.cpp",
        ],
        "//conditions:default": [
            "runtime/src/z_Linux_asm.S",
            "runtime/src/z_Linux_util.cpp",
            "runtime/src/kmp_gsupport.cpp",
        ],
    }),
    hdrs = [
        "runtime/src/kmp.h",
        "runtime/src/kmp_atomic.h",
        "runtime/src/kmp_debug.h",
        "runtime/src/kmp_i18n.h",
        "runtime/src/kmp_io.h",
        "runtime/src/kmp_itt.h",
        "runtime/src/kmp_itt.inl",
        "runtime/src/kmp_lock.h",
        "runtime/src/kmp_os.h",
        "runtime/src/kmp_str.h",
        "runtime/src/kmp_version.h",
        "runtime/src/kmp_wait_release.h",
        "runtime/src/kmp_wrapper_getpid.h",
        "runtime/src/omp.h",
        "runtime/src/ompt-specific.h",
        "runtime/src/thirdparty/ittnotify/ittnotify_config.h",
    ],
    copts = ["-w"],
    data = [
        ":kmp_config_gen",
        ":kmp_i18n_default",
        ":kmp_i18n_id",
        ":omp_gen",
        ":omp_tools_gen",
    ],
    include_prefix = "runtime/src",
    includes = ["runtime/src"],
    local_defines = select({
        "@bazel_tools//src/conditions:windows": [
            "_M_AMD64",
            "OMPT_SUPPORT=0",
        ],
        "//conditions:default": [],
    }),
)
