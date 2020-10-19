load("@rules_cc//cc:defs.bzl", "cc_library")
load("@com_intel_plaidml//bzl:template.bzl", "template_rule")
load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")

package(default_visibility = ["//visibility:public"])

TAGS = [
    "skip_macos",
    "skip_windows",
]

filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

genrule(
    name = "benchmarkapp",
    srcs = [
        ":all",
        "@//plaidml",
    ],
    outs = [
        "benchmark_app",
        "libtbb.so.2",
        "libMKLDNNPlugin.so",
        "libPlaidMLPlugin.so",
    ],
    cmd = """
mkdir -p $(@D)/buildov;
pushd $(@D)/buildov; 
cmake -GNinja ../../../../../../external/openvino -DENABLE_CLDNN=OFF -DENABLE_VPU=OFF -DENABLE_MYRIAD=OFF -DENABLE_MKL_DNN=ON -DENABLE_PLAIDML=ON -DENABLE_SPEECH_DEMO=OFF -DPLAIDML_SRC_PATH=/home/brian/plaidml;
ninja ;
popd ;
cp external/openvino/bin/intel64/Release/benchmark_app $(@D); 
cp external/openvino/bin/intel64/Release/lib/libMKLDNNPlugin.so $(@D); 
cp external/openvino/bin/intel64/Release/lib/libPlaidMLPlugin.so $(@D); 
cp external/openvino/inference-engine/temp/tbb/lib/libtbb.so.2 $(@D);""",
    local = True,
)

"""
cmake_external(
   name = "openvino",
   # TODO - add select for debug, release, etc
   cache_entries = {
       "ENABLE_CLDNN": "OFF",
       "ENABLE_MKL_DNN": "OFF",
       "ENABLE_VPU": "OFF",
       "ENABLE_MYRIAD": "OFF",
       "ENABLE_PLAIDML": "ON",
       "ENABLE_SPEECH_DEMO": "OFF",
       "NGRAPH_IE_ENABLE": "ON",
       "PLAIDML_SRC_PATH": "/home/brian/plaidml",
   },
   deps = [
       "@//plaidml:sdklib"
   ],
   env_vars = {
       "CXXFLAGS": "-U_FORTIFY_SOURCE -w"
   },
   #generate_crosstool_file = True,
   make_commands = [
       "ninja",
       "ls"
   ],
   cmake_options = ["-GNinja"],
   lib_source = "@openvino//:all",
   static_libraries = ["libopenblas.a"],
)
"""
