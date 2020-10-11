load("@com_intel_plaidml//vendor/llvm:llvm.bzl", "cmake_var_string", "expand_cmake_vars", "llvm_defines")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Define OpenMP-specific CMake vars
openmp_cmake_vars = {
    "LIBOMP_LIB_FILE": "libomp.so",
    "LIBOMP_VERSION_MAJOR": "5",
    "LIBOMP_VERSION_MINOR": "0",
    "LIBOMP_VERSION_BUILD": "20140926",
    "LIBOMP_BUILD_DATE": "20140926",
}

openmp_all_cmake_vars = cmake_var_string(openmp_cmake_vars)

# Auto-generated files (generated using configure_file in CMake)
expand_cmake_vars(
    name = "omp_gen",
    src = "runtime/src/include/omp.h.var",
    cmake_vars = openmp_all_cmake_vars,
    dst = "runtime/src/omp.h",
)

expand_cmake_vars(
    name = "kmp_config_gen",
    src = "runtime/src/kmp_config.h.cmake",
    cmake_vars = openmp_all_cmake_vars,
    dst = "runtime/src/kmp_config.h",
)

expand_cmake_vars(
    name = "omp_tools_gen",
    src = "runtime/src/include/omp-tools.h.var",
    cmake_vars = openmp_all_cmake_vars,
    dst = "runtime/src/omp-tools.h",
)

genrule(
    name = "kmp_i18n_id",
    srcs = ["runtime/src/i18n/en_US.txt"],
    outs = ["runtime/src/kmp_i18n_id.inc"],
    cmd = select({
        "@bazel_tools//src/conditions:darwin": """$(location @com_intel_plaidml_conda//:perl) \
            $(location runtime/tools/message-converter.pl) \
            --os=mac \
            --prefix=kmp_i18n \
            --enum=$@ \
            $(location runtime/src/i18n/en_US.txt)
        """,
        "@bazel_tools//src/conditions:windows": """$(location @com_intel_plaidml_conda//:perl) \
            $(location runtime/tools/message-converter.pl) \
            --os=win \
            --prefix=kmp_i18n \
            --enum=$@ \
            $(location runtime/src/i18n/en_US.txt)
        """,
        "//conditions:default": """$(location @com_intel_plaidml_conda//:perl) \
            $(location runtime/tools/message-converter.pl) \
            --os=lin \
            --prefix=kmp_i18n \
            --enum=$@ \
            $(location runtime/src/i18n/en_US.txt)
        """,
    }),
    cmd_ps = """$(location @com_intel_plaidml_conda//:perl) \
        $(location runtime/tools/message-converter.pl) \
        --os=win \
        --prefix=kmp_i18n \
        --enum=$@ \
        $(location runtime/src/i18n/en_US.txt)
    """,
    tools = [
        "runtime/tools/message-converter.pl",
        "@com_intel_plaidml_conda//:perl",
    ],
)

genrule(
    name = "kmp_i18n_default",
    srcs = ["runtime/src/i18n/en_US.txt"],
    outs = ["runtime/src/kmp_i18n_default.inc"],
    cmd = select({
        "@bazel_tools//src/conditions:darwin": """$(location @com_intel_plaidml_conda//:perl) \
            $(location runtime/tools/message-converter.pl) \
            --os=mac \
            --prefix=kmp_i18n \
            --default=$@ \
            $(location runtime/src/i18n/en_US.txt)
        """,
        "@bazel_tools//src/conditions:windows": """$(location @com_intel_plaidml_conda//:perl) \
            $(location runtime/tools/message-converter.pl) \
            --os=win \
            --prefix=kmp_i18n \
            --default=$@ \
            $(location runtime/src/i18n/en_US.txt)
        """,
        "//conditions:default": """$(location @com_intel_plaidml_conda//:perl) \
            $(location runtime/tools/message-converter.pl) \
            --os=lin \
            --prefix=kmp_i18n \
            --default=$@ \
            $(location runtime/src/i18n/en_US.txt)
        """,
    }),
    cmd_ps = """$(location @com_intel_plaidml_conda//:perl) \
        $(location runtime/tools/message-converter.pl) \
        --os=win \
        --prefix=kmp_i18n \
        --default=$@ \
        $(location runtime/src/i18n/en_US.txt)
    """,
    tools = [
        "runtime/tools/message-converter.pl",
        "@com_intel_plaidml_conda//:perl",
    ],
)

filegroup(
    name = "asm",
    srcs = select({
        "@bazel_tools//src/conditions:windows": ["runtime/src/z_Windows_NT-586_asm.asm"],
        "//conditions:default": ["runtime/src/z_Linux_asm.S"],
    }),
)

cc_library(
    name = "openmp",
    srcs = [
        ":asm",
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
            "runtime/src/z_Windows_NT-586_util.cpp",
        ],
        "//conditions:default": [
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
        "runtime/src/ompt-specific.h",
        "runtime/src/thirdparty/ittnotify/ittnotify_config.h",
    ],
    data = [
        ":kmp_config_gen",
        ":kmp_i18n_default",
        ":kmp_i18n_id",
        ":omp_gen",
        ":omp_tools_gen",
    ],
    defines = llvm_defines,
    include_prefix = "runtime/src",
    includes = ["runtime/src"],
)
