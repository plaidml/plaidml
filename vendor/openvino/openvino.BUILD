LOAD("@RULES_CC//CC:DEFS.BZL", "CC_LIBRARY")
LOAD("@COM_INTEL_PLAIDML//BZL:TEMPLATE.BZL", "TEMPLATE_RULE")

PACKAGE(DEFAULT_VISIBILITY = ["//VISIBILITY:PUBLIC"])

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
    name = "cmake",
    srcs = [
        ":all",
        "@//plaidml:sdk_tar",
    ],
    outs = [
        "build.ninja",
    ],
    cmd = """
mkdir -p $(@D)/buildov &&
tar xf $(location @//plaidml:sdk_tar) -C $(@D)/buildov &&
pushd $(@D)/buildov &&
cmake -GNinja ../../../../../../external/openvino -DENABLE_CLDNN=OFF \
 -DENABLE_VPU=OFF -DENABLE_MYRIAD=OFF -DENABLE_MKL_DNN=OFF -DENABLE_GNA=OFF \
 -DENABLE_PLAIDML=ON -DENABLE_SPEECH_DEMO=OFF -DPLAIDML_SO_PATH=./libplaidml.so
popd &&
cp $(@D)/buildov/build.ninja $(@D)
""",
    local = True,
)

genrule(
    name = "benchmark_app_gen",
    srcs = [
        ":all",
        "@//plaidml:sdk_tar",
        ":cmake",
    ],
    outs = [
        "benchmark_app",
        "libtbb.so.2",
        "libplaidml.so",
        "libPlaidMLPlugin.so",
    ],
    cmd = """
pushd $(@D)/buildov &&
ninja &&
popd &&
cp $(@D)/buildov/libplaidml.so $(@D) &&
cp external/openvino/bin/intel64/Release/benchmark_app $(@D) &&
cp external/openvino/bin/intel64/Release/lib/libPlaidMLPlugin.so $(@D) &&
cp external/openvino/inference-engine/temp/tbb/lib/libtbb.so.2 $(@D)""",
    local = True,
)
