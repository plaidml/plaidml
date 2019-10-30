package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

cc_library(
    name = "tbb",
    srcs = glob([
        "tbb/**/*.cpp",
        "tbb/**/*.cc",
        "tbbmalloc/**/*.cpp",
        "tbbmalloc/**/*.cc",
    ]),
    hdrs = glob([
        "include/serial/**",
        "include/tbb/**/**",
    ]),
    includes = ["include"],
)

### TODO: This is only for testing the parallel execution.
### Delete this when the linking of TBB is fully integrated
### to be built natively with Bazel.
# genrule(
#   name = "build_tbb",
#   srcs = glob(["**"]) + [
#     "@local_config_cc//:toolchain",
#   ],
#   cmd = """
#         # set -e
#         WORK_DIR=$$PWD
#         DEST_DIR=$$PWD/$(@D)
#
#         cd $$(dirname $(location :Makefile))
#
#          #TBB's build needs some help to figure out what compiler it's using
#          #if $$CXX --version | grep clang &> /dev/null; then
#         COMPILER_OPT="compiler=clang"
#          #else
#          #  COMPILER_OPT="compiler=gcc"
#          #fi
#
#          # uses extra_inc=big_iron.inc to specify that static libraries are
#          # built. See https://software.intel.com/en-us/forums/intel-threading-building-blocks/topic/297792
#         make tbb_build_prefix="build" \
#             extra_inc=big_iron.inc \
#             $$COMPILER_OPT; \
#
#         echo cp build/build_{release,debug}/*.a $$DEST_DIR
#         cp build/build_{release,debug}/*.a $$DEST_DIR
#         cd $$WORK_DIR
#   """,
#   outs = [
#     "libtbb.a",
#     "libtbbmalloc.a",
#   ]
# )
#
# cc_library(
#     name = "tbb",
#     hdrs = glob([
#         "include/serial/**",
#         "include/tbb/**/**",
#         ]),
#     srcs = ["libtbb.a"],
#     includes = ["include"],
#     visibility = ["//visibility:public"],
# )
