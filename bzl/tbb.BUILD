load("@bazel_skylib//rules:write_file.bzl", "write_file")

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

write_file(
    name = "gen_version_file",
    out = "version_string.ver",
    content = [
        "#define __TBB_VERSION_STRINGS(N) \"Empty\"",
    ],
)

### Generate __TBB_get_cpu_ctl_env and __TBB_set_cpu_ctl_env from C file
write_file(
    name = "gen_cpu_ctl_env",
    out = "gen_cpu_ctl_env.cc",
    content = [
        "",
        "#include <cstddef>",
        "#define private public",
        "#include \"tbb/tbb_machine.h\"",
        "#undef private",
        "const int FE_TONEAREST = 0x0000,",
        "          FE_DOWNWARD = 0x0400,",
        "          FE_UPWARD = 0x0800,",
        "          FE_TOWARDZERO = 0x0c00,",
        "          FE_RND_MODE_MASK = FE_TOWARDZERO,",
        "          SSE_RND_MODE_MASK = FE_RND_MODE_MASK << 3,",
        "          SSE_DAZ = 0x0040,",
        "          SSE_FTZ = 0x8000,",
        "          SSE_MODE_MASK = SSE_DAZ | SSE_FTZ,",
        "          SSE_STATUS_MASK = 0x3F;",
        "const int NumSseModes = 4;",
        "const int SseModes[NumSseModes] = { 0, SSE_DAZ, SSE_FTZ, SSE_DAZ | SSE_FTZ };",
        "#include <float.h>",
        "void __TBB_get_cpu_ctl_env ( tbb::internal::cpu_ctl_env* fe ) {",
        "    fe->x87cw = short(_control87(0, 0) & _MCW_RC) << 2;",
        "    fe->mxcsr = _mm_getcsr();",
        "}",
        "void __TBB_set_cpu_ctl_env ( const tbb::internal::cpu_ctl_env* fe ) {",
        "    _control87( (fe->x87cw & FE_RND_MODE_MASK) >> 6, _MCW_RC );",
        "    _mm_setcsr( fe->mxcsr );",
        "}",
    ],
)

cc_library(
    name = "tbb_win",
    srcs = glob([
               "src/tbb/*.cpp",
               "include/tbb/*.h",
               "src/tbb/*.h",
               "src/rml/include/*.h",
           ]) + ["src/rml/client/rml_tbb.cpp"] +
           select({
               "@com_intel_plaidml//toolchain:windows_x86_64": [
                   ":gen_cpu_ctl_env",
               ],
               "//conditions:default": [],
           }),
    hdrs = [
        ":gen_version_file",
    ],
    defines = [
        "TBB_USE_THREADING_TOOLS",
        "__TBB_DYNAMIC_LOAD_ENABLED=0",
        "__TBB_SOURCE_DIRECTLY_INCLUDED=1",
        "__TBB_x86_64=1",
    ] + select({
        "@com_intel_plaidml//toolchain:windows_x86_64": [
            "USE_WINTHREAD",
            "__TBB_CPU_CTL_ENV_PRESENT=1",
        ],
        "//conditions:default": [
            "USE_PTHREAD",
        ],
    }),
    includes = [
        "build/vs2013",
        "include",
        "src",
        "src/rml/include",
    ],
)

genrule(
    name = "tbb_unix",
    srcs = glob(["**"]),
    outs = [
        "libtbb.a",
        "libtbbmalloc.a",
    ],
    cmd = """
        # set -e
        WORK_DIR=$$PWD
        DEST_DIR=$$PWD/$(@D)
        cd $$(dirname $(location :Makefile))
        #TBB's build needs some help to figure out what compiler it's using
        if $$CXX --version | grep clang &> /dev/null; then
          COMPILER_OPT="compiler=clang"
        else
          COMPILER_OPT="compiler=gcc"
        fi

        # uses extra_inc=big_iron.inc to specify that static libraries are
        # built. See https://software.intel.com/en-us/forums/intel-threading-building-blocks/topic/297792
        make tbb_build_prefix="build" \
            extra_inc=big_iron.inc \
            $$COMPILER_OPT; \

        echo cp build/build_{release,debug}/*.a $$DEST_DIR
        cp build/build_{release,debug}/*.a $$DEST_DIR
        cd $$WORK_DIR
  """,
)

cc_library(
    name = "tbb",
    srcs = select({
        "@com_intel_plaidml//toolchain:windows_x86_64": glob([
            "src/tbb/*.cpp",
            "include/tbb/*.h",
            "src/tbb/*.h",
            "src/rml/include/*.h",
        ]) + [
            "src/rml/client/rml_tbb.cpp",
            ":gen_cpu_ctl_env",
        ],
        "//conditions:default": [
            "libtbb.a",
        ],
    }),
    hdrs = select({
        "@com_intel_plaidml//toolchain:windows_x86_64": [
            ":gen_version_file",
        ],
        "//conditions:default": glob([
            "include/serial/**",
            "include/tbb/**/**",
        ]),
    }),
    defines = select({
        "@com_intel_plaidml//toolchain:windows_x86_64": [
            "TBB_USE_THREADING_TOOLS",
            "__TBB_DYNAMIC_LOAD_ENABLED=0",
            "__TBB_SOURCE_DIRECTLY_INCLUDED=1",
            "__TBB_x86_64=1",
            "USE_WINTHREAD",
            "__TBB_CPU_CTL_ENV_PRESENT=1",
        ],
        "//conditions:default": [],
    }),
    includes = select({
        "@com_intel_plaidml//toolchain:windows_x86_64": [
            "build/vs2013",
            "include",
            "src",
            "src/rml/include",
        ],
        "//conditions:default": [
            "include",
        ],
    }),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
