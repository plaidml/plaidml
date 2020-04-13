# Description:
#   The MLIR "Multi-Level Intermediate Representation" Compiler Infrastructure

load("@com_intel_plaidml//vendor/mlir:tblgen.bzl", "gentbl")

licenses(["notice"])

package(default_visibility = [":friends"])

package_group(
    name = "subpackages",
    packages = ["//..."],
)

package_group(
    name = "friends",
    packages = ["//..."],
)

exports_files([
    "LICENSE.TXT",
    "include/mlir/Dialect/LLVMIR/LLVMOps.td",
    "run_lit.sh",
])

LINKOPTS = select({
    "@bazel_tools//src/conditions:windows": [],
    "//conditions:default": [
        "-lm",
        "-pthread",
    ],
})

cc_library(
    name = "DialectSymbolRegistry",
    # strip_include_prefix does not apply to textual_hdrs.
    hdrs = ["include/mlir/IR/DialectSymbolRegistry.def"],
    strip_include_prefix = "include/mlir/IR",
    textual_hdrs = ["include/mlir/IR/DialectSymbolRegistry.def"],
)

gentbl(
    name = "OpAsmInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "include/mlir/IR/OpAsmInterface.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir/IR/OpAsmInterface.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/IR/OpAsmInterface.td",
    td_srcs = [
        ":OpBaseTdFiles",
    ],
)

cc_library(
    name = "IR",
    srcs = glob([
        "lib/IR/*.cpp",
        "lib/IR/*.h",
    ]),
    hdrs = glob([
        "include/mlir/IR/*.h",
    ]) + [
        "include/mlir/Interfaces/CallInterfaces.h",
    ],
    includes = ["include"],
    deps = [
        ":CallOpInterfacesIncGen",
        ":DialectSymbolRegistry",
        ":InferTypeOpInterfaceIncGen",
        ":OpAsmInterfacesIncGen",
        ":SideEffectInterfacesIncGen",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "Pass",
    srcs = glob([
        "lib/Pass/*.cpp",
        "lib/Pass/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Pass/*.h",
    ]),
    includes = ["include"],
    linkopts = LINKOPTS,
    deps = [
        ":Analysis",
        ":IR",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

# TODO(ntv): Update these to enable simplifying the cmake and build files.
cc_library(
    name = "EDSC",
    srcs = [
        "lib/EDSC/Builders.cpp",
    ],
    hdrs = [
        "include/mlir-c/Core.h",
        "include/mlir/EDSC/Builders.h",
        "include/mlir/EDSC/Intrinsics.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "EDSCInterface",
    srcs = [
        "lib/EDSC/CoreAPIs.cpp",
    ],
    hdrs = [
        "include/mlir-c/Core.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":Parser",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

filegroup(
    name = "OpBaseTdFiles",
    srcs = [
        "include/mlir/IR/OpBase.td",
    ],
)

##---------------------------------------------------------------------------##
# Affine dialect.
##---------------------------------------------------------------------------##

filegroup(
    name = "PassBaseTdFiles",
    srcs = [
        "include/mlir/Pass/PassBase.td",
    ],
)

filegroup(
    name = "AffineOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Affine/IR/AffineOps.td",
        "include/mlir/Dialect/Affine/IR/AffineOpsBase.td",
        "include/mlir/Interfaces/LoopLikeInterface.td",
        "include/mlir/Interfaces/SideEffects.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "AffineOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/Affine/IR/AffineOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/Affine/IR/AffineOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls",
            "include/mlir/Dialect/Affine/IR/AffineOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Affine/IR/AffineOps.td",
    td_srcs = [
        ":AffineOpsTdFiles",
    ],
)

##---------------------------------------------------------------------------##
# AVX512 dialect.
##---------------------------------------------------------------------------##

filegroup(
    name = "AVX512TdFiles",
    srcs = [
        "include/mlir/Dialect/AVX512/AVX512.td",
        "include/mlir/Dialect/LLVMIR/LLVMOpBase.td",
        "include/mlir/IR/OpBase.td",
        "include/mlir/Interfaces/SideEffects.td",
    ],
)

gentbl(
    name = "AVX512IncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-dialect-decls -dialect=avx512",
            "include/mlir/Dialect/AVX512/AVX512Dialect.h.inc",
        ),
        (
            "-gen-op-decls",
            "include/mlir/Dialect/AVX512/AVX512.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/AVX512/AVX512.cpp.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/Dialects/AVX512/AVX512.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/AVX512/AVX512.td",
    td_srcs = [
        ":AVX512TdFiles",
    ],
)

cc_library(
    name = "AVX512",
    srcs = [
        "lib/Dialect/AVX512/IR/AVX512Dialect.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/AVX512/AVX512Dialect.h",
    ],
    includes = ["include"],
    deps = [
        ":AVX512IncGen",
        ":IR",
        ":SideEffects",
        ":VectorOps",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "AVX512ToLLVM",
    srcs = glob([
        "lib/Conversion/AVX512ToLLVM/*.cpp",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/AVX512ToLLVM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":AVX512",
        ":ConversionPassIncGen",
        ":EDSC",
        ":IR",
        ":LLVMAVX512",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":VectorOps",
        ":VectorToLLVM",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
)

filegroup(
    name = "LoopOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/LoopOps/LoopOps.td",
        "include/mlir/Interfaces/LoopLikeInterface.td",
        "include/mlir/Interfaces/SideEffects.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "LoopOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/LoopOps/LoopOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/LoopOps/LoopOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls",
            "include/mlir/Dialect/LoopOps/LoopOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LoopOps/LoopOps.td",
    td_srcs = [
        ":LoopOpsTdFiles",
    ],
)

gentbl(
    name = "LoopPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-pass-decls",
            "include/mlir/Dialect/LoopOps/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LoopOps/Passes.td",
    td_srcs = [
        ":PassBaseTdFiles",
    ],
)

cc_library(
    name = "LoopOpsTransforms",
    srcs = glob([
        "lib/Dialect/LoopOps/Transforms/*.cpp",
        "lib/Dialect/LoopOps/Transforms/*.h",
    ]),
    hdrs = ["include/mlir/Dialect/LoopOps/Passes.h"],
    includes = ["include"],
    deps = [
        ":Affine",
        ":IR",
        ":LoopOps",
        ":LoopPassIncGen",
        ":Pass",
        ":StandardOps",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
)

filegroup(
    name = "StdOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/StandardOps/IR/Ops.td",
        "include/mlir/IR/OpAsmInterface.td",
        "include/mlir/Interfaces/CallInterfaces.td",
        "include/mlir/Interfaces/ControlFlowInterfaces.td",
        "include/mlir/Interfaces/SideEffects.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "StandardOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/StandardOps/IR/Ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/StandardOps/IR/Ops.cpp.inc",
        ),
        (
            "-gen-dialect-decls",
            "include/mlir/Dialect/StandardOps/IR/OpsDialect.h.inc",
        ),
        (
            "-gen-enum-decls",
            "include/mlir/Dialect/StandardOps/IR/OpsEnums.h.inc",
        ),
        (
            "-gen-enum-defs",
            "include/mlir/Dialect/StandardOps/IR/OpsEnums.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/StandardOps/IR/Ops.td",
    td_srcs = [
        ":StdOpsTdFiles",
    ],
)

cc_library(
    name = "Dialect",
    srcs = glob([
        "lib/Dialect/*.cpp",
        "lib/Dialect/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Dialect/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":IR",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "DialectUtils",
    srcs = glob([
        "lib/Dialect/Utils/*.cpp",
        "lib/Dialect/Utils/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Dialect/Utils/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":IR",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "Affine",
    srcs = glob(
        [
            "lib/Dialect/Affine/IR/*.cpp",
            "lib/Dialect/Affine/IR/*.h",
            "lib/Dialect/Affine/EDSC/*.cpp",
        ],
    ) + [
        "include/mlir/Transforms/InliningUtils.h",
    ],
    hdrs = glob([
        "include/mlir/Dialect/Affine/IR/*.h",
        "include/mlir/Dialect/Affine/EDSC/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":AffineOpsIncGen",
        ":EDSC",
        ":IR",
        ":LoopLikeInterface",
        ":SideEffects",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "AffinePassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-pass-decls",
            "include/mlir/Dialect/Affine/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Affine/Passes.td",
    td_srcs = [
        ":PassBaseTdFiles",
    ],
)

cc_library(
    name = "AffineTransforms",
    srcs = glob([
        "lib/Dialect/Affine/Transforms/*.cpp",
        "lib/Dialect/Affine/Transforms/*.h",
    ]),
    hdrs = [
        "include/mlir/Dialect/Affine/Passes.h",
    ],
    includes = ["include"],
    deps = [
        ":Affine",
        ":AffinePassIncGen",
        ":Analysis",
        ":IR",
        ":LoopOps",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":VectorOps",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "ConversionPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-pass-decls",
            "include/mlir/Conversion/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Conversion/Passes.td",
    td_srcs = [
        ":PassBaseTdFiles",
    ],
)

cc_library(
    name = "AffineToStandardTransforms",
    srcs = glob([
        "lib/Conversion/AffineToStandard/*.cpp",
        "lib/Conversion/AffineToStandard/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob(["include/mlir/Conversion/AffineToStandard/*.h"]),
    includes = ["include"],
    deps = [
        ":Affine",
        ":ConversionPassIncGen",
        ":IR",
        ":LoopOps",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
    ],
)

# SDBM dialect only contains attribute components that can be constructed given
# a dialect object, so whenever it is used it must also be registered. Therefore
# we don't split out the registration library for it.
cc_library(
    name = "SDBM",
    srcs = glob([
        "lib/Dialect/SDBM/*.cpp",
        "lib/Dialect/SDBM/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Dialect/SDBM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":IR",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "LoopOps",
    srcs = glob(
        [
            "lib/Dialect/LoopOps/*.cpp",
            "lib/Dialect/LoopOps/*.h",
            "lib/Dialect/LoopOps/EDSC/*.cpp",
        ],
    ),
    hdrs = glob([
        "include/mlir/Dialect/LoopOps/*.h",
        "include/mlir/Dialect/LoopOps/EDSC/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":EDSC",
        ":IR",
        ":LoopLikeInterface",
        ":LoopOpsIncGen",
        ":SideEffects",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "LoopLikeInterface",
    srcs = ["lib/Interfaces/LoopLikeInterface.cpp"],
    hdrs = ["include/mlir/Interfaces/LoopLikeInterface.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":LoopLikeInterfaceIncGen",
    ],
)

gentbl(
    name = "ShapeOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/Shape/IR/ShapeOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/Shape/IR/ShapeOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls",
            "include/mlir/Dialect/Shape/IR/ShapeOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Shape/IR/ShapeOps.td",
    td_srcs = [
        ":StdOpsTdFiles",
        "include/mlir/Interfaces/InferTypeOpInterface.td",
    ],
)

cc_library(
    name = "Shape",
    srcs = glob(
        [
            "lib/Dialect/Shape/IR/*.cpp",
        ],
    ),
    hdrs = glob([
        "include/mlir/Dialect/Shape/IR/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":CallOpInterfaces",
        ":CommonFolders",
        ":IR",
        ":InferTypeOpInterface",
        ":ShapeOpsIncGen",
        ":SideEffects",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "StandardOps",
    srcs = glob(
        [
            "lib/Dialect/StandardOps/IR/*.cpp",
            "lib/Dialect/StandardOps/IR/*.h",
            "lib/Dialect/StandardOps/EDSC/*.cpp",
        ],
    ),
    hdrs = glob([
        "include/mlir/Dialect/StandardOps/IR/*.h",
        "include/mlir/Dialect/StandardOps/EDSC/*.h",
    ]) + [
        "include/mlir/Transforms/InliningUtils.h",
    ],
    includes = ["include"],
    deps = [
        ":CallOpInterfaces",
        ":CommonFolders",
        ":ControlFlowInterfaces",
        ":EDSC",
        ":IR",
        ":SideEffects",
        ":StandardOpsIncGen",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "StandardOpsTransforms",
    srcs = glob(
        [
            "lib/Dialect/StandardOps/Transforms/*.cpp",
            "lib/Dialect/StandardOps/Transforms/*.h",
        ],
    ),
    hdrs = glob([
        "include/mlir/Dialect/StandardOps/Transforms/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":Analysis",
        ":ControlFlowInterfaces",
        ":IR",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "VectorOps",
    srcs = glob(
        [
            "lib/Dialect/Vector/*.cpp",
            "lib/Dialect/Vector/*.h",
            "lib/Dialect/Vector/EDSC/*.cpp",
            "lib/Dialect/Vector/EDSC/*.h",
        ],
    ),
    hdrs = glob([
        "include/mlir/Dialect/Vector/*.h",
        "include/mlir/Dialect/Vector/EDSC/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":Affine",
        ":Analysis",
        ":DialectUtils",
        ":EDSC",
        ":IR",
        ":SideEffects",
        ":StandardOps",
        ":Support",
        ":VectorOpsIncGen",
        ":VectorTransformPatternsIncGen",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "Support",
    srcs = glob(
        [
            "lib/Support/*.cpp",
            "lib/Support/*.h",
        ],
        exclude = [
            # TODO(herhut): Move JitRunner out of Support so that Support does not
            # depend on dialect.
            "lib/Support/JitRunner.cpp",
            # TODO(jpienaar): Move this out, else Support depends on Analysis/
            "lib/Support/MlirOptMain.cpp",
        ],
    ),
    hdrs = glob([
        "include/mlir/ADT/*.h",
        "include/mlir/Support/*.h",
    ]),
    includes = ["include"],
    deps = [
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "ParserTokenKinds",
    # strip_include_prefix does not apply to textual_hdrs.
    hdrs = ["lib/Parser/TokenKinds.def"],
    strip_include_prefix = "lib/Parser",
    textual_hdrs = ["lib/Parser/TokenKinds.def"],
)

cc_library(
    name = "Parser",
    srcs = glob([
        "lib/Parser/*.cpp",
        "lib/Parser/*.h",
    ]),
    hdrs = [
        "include/mlir/Parser.h",
    ],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
        ":ParserTokenKinds",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

filegroup(
    name = "LLVMAVX512TdFiles",
    srcs = [
        "include/mlir/Dialect/LLVMIR/LLVMAVX512.td",
        ":LLVMOpsTdFiles",
    ],
)

gentbl(
    name = "LLVMAVX512IncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-dialect-decls -dialect=llvm_avx512",
            "include/mlir/Dialect/LLVMIR/LLVMAVX512Dialect.h.inc",
        ),
        (
            "-gen-op-decls",
            "include/mlir/Dialect/LLVMIR/LLVMAVX512.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/LLVMIR/LLVMAVX512.cpp.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/Dialects/LLVMIR/LLVMAVX512.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/LLVMAVX512.td",
    td_srcs = [
        ":LLVMAVX512TdFiles",
    ],
)

cc_library(
    name = "LLVMAVX512",
    srcs = [
        "lib/Dialect/LLVMIR/IR/LLVMAVX512Dialect.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/LLVMIR/LLVMAVX512Dialect.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMAVX512IncGen",
        ":LLVMDialect",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "LLVMAVX512ConversionIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-llvmir-conversions",
            "include/mlir/Dialect/LLVMIR/LLVMAVX512Conversions.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/LLVMAVX512.td",
    td_srcs = [
        ":LLVMAVX512TdFiles",
    ],
)

cc_library(
    name = "TargetLLVMAVX512Intr",
    srcs = [
        "lib/Target/LLVMIR/LLVMAVX512Intr.cpp",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMAVX512",
        ":LLVMAVX512ConversionIncGen",
        ":LLVMIRModuleTranslation",
        ":Translation",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "LLVMDialect",
    srcs = glob(
        [
            "lib/Dialect/LLVMIR/IR/*.cpp",
            "lib/Dialect/LLVMIR/IR/*.h",
        ],
        exclude = [
            "lib/Dialect/LLVMIR/IR/*AVX512*.cpp",
            "lib/Dialect/LLVMIR/IR/*AVX512*.h",
            "lib/Dialect/LLVMIR/IR/NVVM*.cpp",
            "lib/Dialect/LLVMIR/IR/NVVM*.h",
            "lib/Dialect/LLVMIR/IR/ROCDL*.cpp",
            "lib/Dialect/LLVMIR/IR/ROCDL*.h",
        ],
    ),
    hdrs = glob(
        [
            "include/mlir/Dialect/LLVMIR/*.h",
        ],
        exclude = [
            "include/mlir/Dialect/LLVMIR/*AVX512*.h",
            "include/mlir/Dialect/LLVMIR/NVVM*.h",
            "include/mlir/Dialect/LLVMIR/ROCDL*.h",
        ],
    ),
    includes = ["include"],
    deps = [
        ":ControlFlowInterfaces",
        ":IR",
        ":LLVMOpsIncGen",
        ":SideEffects",
        ":Support",
        "@llvm-project//llvm:asm_parser",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "LLVMPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-pass-decls",
            "include/mlir/Dialect/LLVMIR/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/Transforms/Passes.td",
    td_srcs = [
        ":PassBaseTdFiles",
    ],
)

cc_library(
    name = "LLVMIRTransforms",
    srcs = glob([
        "lib/Dialect/LLVMIR/Transforms/*.cpp",
        "lib/Dialect/LLVMIR/Transforms/*.h",
    ]),
    hdrs = glob(["include/mlir/Dialect/LLVMIR/Transforms/*.h"]),
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":LLVMPassIncGen",
        ":Pass",
    ],
)

filegroup(
    name = "GPUOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/GPU/GPUBase.td",
        "include/mlir/Dialect/GPU/GPUOps.td",
        "include/mlir/Dialect/LLVMIR/LLVMOpBase.td",
        "include/mlir/Interfaces/SideEffects.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "ParallelLoopMapperAttrGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-struct-attr-decls",
            "include/mlir/Dialect/GPU/ParallelLoopMapperAttr.h.inc",
        ),
        (
            "-gen-struct-attr-defs",
            "include/mlir/Dialect/GPU/ParallelLoopMapperAttr.cpp.inc",
        ),
        (
            "-gen-enum-decls",
            "include/mlir/Dialect/GPU/ParallelLoopMapperEnums.h.inc",
        ),
        (
            "-gen-enum-defs",
            "include/mlir/Dialect/GPU/ParallelLoopMapperEnums.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/GPU/ParallelLoopMapperAttr.td",
    td_srcs = [
        ":GPUOpsTdFiles",
        ":AffineOpsTdFiles",
    ],
)

gentbl(
    name = "GPUOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/GPU/GPUOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/GPU/GPUOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls -dialect=gpu",
            "include/mlir/Dialect/GPU/GPUOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/GPU/GPUOps.td",
    td_srcs = [
        ":GPUOpsTdFiles",
    ],
)

cc_library(
    name = "GPUDialect",
    srcs = glob(
        [
            "lib/Dialect/GPU/IR/*.cpp",
            "lib/Dialect/GPU/IR/*.h",
        ],
    ),
    hdrs = glob([
        "include/mlir/Dialect/GPU/GPUDialect.h",
    ]),
    includes = ["include"],
    deps = [
        ":GPUOpsIncGen",
        ":IR",
        ":LLVMDialect",
        ":SideEffects",
        ":StandardOps",
        ":Support",
    ],
)

gentbl(
    name = "GPUPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-pass-decls",
            "include/mlir/Dialect/GPU/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/GPU/Passes.td",
    td_srcs = [
        ":PassBaseTdFiles",
    ],
)

cc_library(
    name = "GPUTransforms",
    srcs = glob(
        [
            "lib/Dialect/GPU/Transforms/*.cpp",
            "lib/Dialect/GPU/Transforms/*.h",
        ],
    ),
    hdrs = [
        "include/mlir/Dialect/GPU/MemoryPromotion.h",
        "include/mlir/Dialect/GPU/ParallelLoopMapper.h",
        "include/mlir/Dialect/GPU/Passes.h",
        "include/mlir/Dialect/GPU/Utils.h",
    ],
    includes = ["include"],
    deps = [
        ":EDSC",
        ":GPUDialect",
        ":GPUPassIncGen",
        ":IR",
        ":LoopOps",
        ":ParallelLoopMapperAttrGen",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
)

filegroup(
    name = "LLVMOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/LLVMIR/LLVMOpBase.td",
        "include/mlir/Dialect/LLVMIR/LLVMOps.td",
        "include/mlir/Interfaces/ControlFlowInterfaces.td",
        "include/mlir/Interfaces/SideEffects.td",
        ":OpBaseTdFiles",
    ],
)

cc_library(
    name = "GPUCommonTransforms",
    hdrs = [
        "lib/Conversion/GPUCommon/IndexIntrinsicsOpLowering.h",
        "lib/Conversion/GPUCommon/OpToFuncCallLowering.h",
    ],
    deps = [
        ":GPUDialect",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":StandardOps",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "GPUToNVVMGen",
    strip_include_prefix = "lib/Conversion/GPUToNVVM",
    tbl_outs = [
        (
            "-gen-rewriters",
            "lib/Conversion/GPUToNVVM/GPUToNVVM.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "lib/Conversion/GPUToNVVM/GPUToNVVM.td",
    td_srcs = [
        ":GPUOpsTdFiles",
        ":NVVMOpsTdFiles",
    ],
)

cc_library(
    name = "GPUToNVVMTransforms",
    srcs = glob([
        "lib/Conversion/GPUToNVVM/*.cpp",
        "lib/Conversion/GPUToNVVM/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/GPUToNVVM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":GPUCommonTransforms",
        ":GPUDialect",
        ":GPUToNVVMGen",
        ":GPUTransforms",
        ":IR",
        ":LLVMTransforms",
        ":NVVMDialect",
        ":Pass",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "GPUToROCDLTransforms",
    srcs = [
        "lib/Conversion/GPUToROCDL/LowerGpuOpsToROCDLOps.cpp",
        "lib/Conversion/PassDetail.h",
    ],
    hdrs = [
        "include/mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h",
    ],
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":GPUCommonTransforms",
        ":GPUDialect",
        ":LLVMTransforms",
        ":Pass",
        ":ROCDLDialect",
        ":Transforms",
    ],
)

cc_library(
    name = "GPUToVulkanTransforms",
    srcs = [
        "lib/Conversion/GPUToVulkan/ConvertGPULaunchFuncToVulkanLaunchFunc.cpp",
        "lib/Conversion/GPUToVulkan/ConvertLaunchFuncToVulkanCalls.cpp",
        "lib/Conversion/PassDetail.h",
    ],
    hdrs = ["include/mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"],
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":GPUDialect",
        ":IR",
        ":LLVMDialect",
        ":Pass",
        ":SPIRVDialect",
        ":SPIRVSerialization",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "GPUToCUDATransforms",
    srcs = [
        "lib/Conversion/GPUToCUDA/ConvertKernelFuncToCubin.cpp",
        "lib/Conversion/GPUToCUDA/ConvertLaunchFuncToCudaCalls.cpp",
        "lib/Conversion/PassDetail.h",
    ],
    hdrs = ["include/mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"],
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":GPUDialect",
        ":IR",
        ":LLVMDialect",
        ":Pass",
        ":Support",
        ":TargetNVVMIR",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:nvptx_code_gen",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:target",
    ],
)

gentbl(
    name = "GPUToSPIRVIncGen",
    strip_include_prefix = "lib/Conversion/GPUToSPIRV",
    tbl_outs = [
        (
            "-gen-rewriters",
            "lib/Conversion/GPUToSPIRV/GPUToSPIRV.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "lib/Conversion/GPUToSPIRV/GPUToSPIRV.td",
    td_srcs = [
        ":GPUOpsTdFiles",
        ":SPIRVOpsTdFiles",
    ],
)

cc_library(
    name = "GPUToSPIRVTransforms",
    srcs = [
        "lib/Conversion/GPUToSPIRV/ConvertGPUToSPIRV.cpp",
        "lib/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.cpp",
        "lib/Conversion/PassDetail.h",
    ],
    hdrs = [
        "include/mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRV.h",
        "include/mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h",
    ],
    includes = [
        "include",
        "lib/Conversions/GPUToSPIRV",
    ],
    deps = [
        ":ConversionPassIncGen",
        ":GPUDialect",
        ":GPUToSPIRVIncGen",
        ":IR",
        ":LoopOps",
        ":Pass",
        ":SPIRVDialect",
        ":SPIRVLowering",
        ":StandardToSPIRVConversions",
        ":Support",
        ":Transforms",
    ],
)

gentbl(
    name = "LLVMOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/LLVMIR/LLVMOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/LLVMIR/LLVMOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls",
            "include/mlir/Dialect/LLVMIR/LLVMOpsDialect.h.inc",
        ),
        (
            "-gen-enum-decls",
            "include/mlir/Dialect/LLVMIR/LLVMOpsEnums.h.inc",
        ),
        (
            "-gen-enum-defs",
            "include/mlir/Dialect/LLVMIR/LLVMOpsEnums.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/LLVMOps.td",
    td_srcs = [
        ":LLVMOpsTdFiles",
    ],
)

gentbl(
    name = "LLVMConversionIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-llvmir-conversions",
            "include/mlir/Dialect/LLVMIR/LLVMConversions.inc",
        ),
        (
            "-gen-enum-to-llvmir-conversions",
            "include/mlir/Dialect/LLVMIR/LLVMConversionEnumsToLLVM.inc",
        ),
        (
            "-gen-enum-from-llvmir-conversions",
            "include/mlir/Dialect/LLVMIR/LLVMConversionEnumsFromLLVM.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/LLVMOps.td",
    td_srcs = [
        ":LLVMOpsTdFiles",
    ],
)

cc_library(
    name = "NVVMDialect",
    srcs = [
        "lib/Dialect/LLVMIR/IR/NVVMDialect.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/LLVMIR/NVVMDialect.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":NVVMOpsIncGen",
        ":SideEffects",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:asm_parser",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
)

filegroup(
    name = "NVVMOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/LLVMIR/LLVMOpBase.td",
        "include/mlir/Dialect/LLVMIR/NVVMOps.td",
        "include/mlir/Interfaces/SideEffects.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "NVVMOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/LLVMIR/NVVMOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/LLVMIR/NVVMOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls -dialect=nvvm",
            "include/mlir/Dialect/LLVMIR/NVVMOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/NVVMOps.td",
    td_srcs = [
        ":NVVMOpsTdFiles",
    ],
)

gentbl(
    name = "NVVMConversionIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-llvmir-conversions",
            "include/mlir/Dialect/LLVMIR/NVVMConversions.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/NVVMOps.td",
    td_srcs = [
        ":NVVMOpsTdFiles",
    ],
)

cc_library(
    name = "ROCDLDialect",
    srcs = [
        "lib/Dialect/LLVMIR/IR/ROCDLDialect.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/LLVMIR/ROCDLDialect.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":ROCDLOpsIncGen",
        ":SideEffects",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:asm_parser",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
)

filegroup(
    name = "ROCDLOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/LLVMIR/LLVMOpBase.td",
        "include/mlir/Dialect/LLVMIR/ROCDLOps.td",
        "include/mlir/Interfaces/SideEffects.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "ROCDLOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/LLVMIR/ROCDLOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/LLVMIR/ROCDLOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls -dialect=rocdl",
            "include/mlir/Dialect/LLVMIR/ROCDLOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/ROCDLOps.td",
    td_srcs = [
        ":ROCDLOpsTdFiles",
    ],
)

gentbl(
    name = "ROCDLConversionIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-llvmir-conversions",
            "include/mlir/Dialect/LLVMIR/ROCDLConversions.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/ROCDLOps.td",
    td_srcs = [
        ":ROCDLOpsTdFiles",
    ],
)

# TODO(gcmn): Update SPIRV dependencies so that they map better to cmake files.
filegroup(
    name = "SPIRVOpsTdFiles",
    srcs = [
        "include/mlir/Interfaces/CallInterfaces.td",
        "include/mlir/Interfaces/ControlFlowInterfaces.td",
        "include/mlir/Interfaces/SideEffects.td",
        ":OpBaseTdFiles",
    ] + glob(["include/mlir/Dialect/SPIRV/*.td"]),
)

gentbl(
    name = "SPIRVOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/SPIRV/SPIRVOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/SPIRV/SPIRVOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls",
            "include/mlir/Dialect/SPIRV/SPIRVOpsDialect.h.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/Dialects/SPIRV/SPIRVOps.md",
        ),
        (
            "-gen-enum-decls",
            "include/mlir/Dialect/SPIRV/SPIRVEnums.h.inc",
        ),
        (
            "-gen-enum-defs",
            "include/mlir/Dialect/SPIRV/SPIRVEnums.cpp.inc",
        ),
        (
            "-gen-spirv-enum-avail-decls",
            "include/mlir/Dialect/SPIRV/SPIRVEnumAvailability.h.inc",
        ),
        (
            "-gen-spirv-enum-avail-defs",
            "include/mlir/Dialect/SPIRV/SPIRVEnumAvailability.cpp.inc",
        ),
        (
            "-gen-spirv-capability-implication",
            "include/mlir/Dialect/SPIRV/SPIRVCapabilityImplication.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/SPIRVOps.td",
    td_srcs = [
        ":SPIRVOpsTdFiles",
    ],
)

gentbl(
    name = "SPIRVCanonicalizationIncGen",
    strip_include_prefix = "lib/Dialect/SPIRV",
    tbl_outs = [
        (
            "-gen-rewriters",
            "lib/Dialect/SPIRV/SPIRVCanonicalization.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "lib/Dialect/SPIRV/SPIRVCanonicalization.td",
    td_srcs = [
        ":SPIRVOpsTdFiles",
        "lib/Dialect/SPIRV/SPIRVCanonicalization.td",
    ],
)

gentbl(
    name = "SPIRVAvailabilityIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-avail-interface-decls",
            "include/mlir/Dialect/SPIRV/SPIRVAvailability.h.inc",
        ),
        (
            "-gen-avail-interface-defs",
            "include/mlir/Dialect/SPIRV/SPIRVAvailability.cpp.inc",
        ),
        (
            "-gen-spirv-avail-impls",
            "include/mlir/Dialect/SPIRV/SPIRVOpAvailabilityImpl.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/SPIRVOps.td",
    td_srcs = [
        ":SPIRVOpsTdFiles",
    ],
)

gentbl(
    name = "SPIRVTargetAndABIStructGen",
    tbl_outs = [
        (
            "-gen-struct-attr-decls",
            "include/mlir/Dialect/SPIRV/TargetAndABI.h.inc",
        ),
        (
            "-gen-struct-attr-defs",
            "include/mlir/Dialect/SPIRV/TargetAndABI.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/TargetAndABI.td",
    td_srcs = [
        ":SPIRVOpsTdFiles",
        ":StdOpsTdFiles",
    ],
)

gentbl(
    name = "SPIRVOpUtilsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-spirv-op-utils",
            "include/mlir/Dialect/SPIRV/SPIRVOpUtils.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/SPIRVBase.td",
    td_srcs = [
        ":SPIRVOpsTdFiles",
        ":SPIRVAvailabilityIncGen",
    ],
)

gentbl(
    name = "SPIRVSerializationGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-spirv-serialization",
            "include/mlir/Dialect/SPIRV/SPIRVSerialization.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/SPIRVOps.td",
    td_srcs = [
        ":SPIRVOpsTdFiles",
    ],
)

cc_library(
    name = "SPIRVDialect",
    srcs = glob(
        [
            "lib/Dialect/SPIRV/*.cpp",
            "lib/Dialect/SPIRV/*.h",
        ],
        exclude = [
            "lib/Dialect/SPIRV/SPIRVLowering.cpp",
        ],
    ) + [
        "include/mlir/Transforms/InliningUtils.h",
    ],
    hdrs = glob(
        [
            "include/mlir/Dialect/SPIRV/*.h",
        ],
        exclude = [
            "include/mlir/Dialect/SPIRV/SPIRVBinaryUtils.h",
            "include/mlir/Dialect/SPIRV/SPIRVLowering.h",
        ],
    ),
    includes = ["include"],
    deps = [
        ":CommonFolders",
        ":ControlFlowInterfaces",
        ":IR",
        ":Parser",
        ":Pass",
        ":SPIRVAvailabilityIncGen",
        ":SPIRVCanonicalizationIncGen",
        ":SPIRVOpUtilsIncGen",
        ":SPIRVOpsIncGen",
        ":SPIRVSerializationGen",
        ":SPIRVTargetAndABIStructGen",
        ":SideEffects",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "SPIRVPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-pass-decls",
            "include/mlir/Dialect/SPIRV/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/Passes.td",
    td_srcs = [
        ":PassBaseTdFiles",
    ],
)

cc_library(
    name = "SPIRVLowering",
    srcs = glob([
        "lib/Dialect/SPIRV/Transforms/*.cpp",
        "lib/Dialect/SPIRV/Transforms/*.h",
    ]) + [
        "lib/Dialect/SPIRV/SPIRVLowering.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/SPIRV/Passes.h",
        "include/mlir/Dialect/SPIRV/SPIRVLowering.h",
        "include/mlir/Dialect/SPIRV/TargetAndABI.h",
    ],
    includes = [
        "include",
    ],
    deps = [
        ":IR",
        ":Pass",
        ":SPIRVDialect",
        ":SPIRVPassIncGen",
        ":SPIRVTargetAndABIStructGen",
        ":StandardOps",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "StandardToSPIRVConversions",
    srcs = glob([
        "lib/Conversion/StandardToSPIRV/*.cpp",
        "lib/Conversion/StandardToSPIRV/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/StandardToSPIRV/*.h",
    ]),
    includes = [
        "include",
        "lib/Conversion/StandardToSPIRV",
    ],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":Pass",
        ":SPIRVDialect",
        ":SPIRVLowering",
        ":StandardOps",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "StandardToStandard",
    srcs = glob([
        "lib/Conversion/StandardToStandard/*.cpp",
        "lib/Conversion/StandardToStandard/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Conversion/StandardToStandard/*.h",
    ]),
    includes = [
        "include",
        "lib/Conversion/StandardToStandard",
    ],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":Pass",
        ":StandardOps",
        ":Transforms",
    ],
)

cc_library(
    name = "SPIRVSerialization",
    srcs = glob(
        [
            "lib/Dialect/SPIRV/Serialization/*.cpp",
        ],
        exclude = [
            "lib/Dialect/SPIRV/Serialization/TranslateRegistration.cpp",
        ],
    ),
    hdrs = [
        "include/mlir/Dialect/SPIRV/SPIRVBinaryUtils.h",
        "include/mlir/Dialect/SPIRV/Serialization.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":SPIRVDialect",
        ":SPIRVOpUtilsIncGen",
        ":SPIRVOpsIncGen",
        ":SPIRVSerializationGen",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "SPIRVTranslateRegistration",
    srcs = [
        "lib/Dialect/SPIRV/Serialization/TranslateRegistration.cpp",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":Parser",
        ":SPIRVDialect",
        ":SPIRVSerialization",
        ":Support",
        ":Translation",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "TransformUtils",
    srcs = glob([
        "lib/Transforms/Utils/*.cpp",
        "lib/Transforms/Utils/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Transforms/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":Affine",
        ":Analysis",
        ":ControlFlowInterfaces",
        ":IR",
        ":LoopLikeInterface",
        ":LoopOps",
        ":SideEffects",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "DerivedAttributeOpInterfaceIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "include/mlir/Interfaces/DerivedAttributeOpInterface.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir/Interfaces/DerivedAttributeOpInterface.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/DerivedAttributeOpInterface.td",
    td_srcs = [
        ":OpBaseTdFiles",
    ],
)

cc_library(
    name = "DerivedAttributeOpInterface",
    srcs = [
        "lib/Interfaces/DerivedAttributeOpInterface.cpp",
    ],
    hdrs = [
        "include/mlir/Interfaces/DerivedAttributeOpInterface.h",
    ],
    includes = ["include"],
    deps = [
        ":DerivedAttributeOpInterfaceIncGen",
        ":IR",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "LoopLikeInterfaceIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "include/mlir/Interfaces/LoopLikeInterface.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir/Interfaces/LoopLikeInterface.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/LoopLikeInterface.td",
    td_srcs = [
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "TransformsPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-pass-decls",
            "include/mlir/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Transforms/Passes.td",
    td_srcs = [
        ":PassBaseTdFiles",
    ],
)

cc_library(
    name = "Transforms",
    srcs = glob([
        "lib/Transforms/*.cpp",
        "lib/Transforms/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Transforms/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":Affine",
        ":Analysis",
        ":IR",
        ":LoopLikeInterface",
        ":LoopOps",
        ":Pass",
        ":SideEffects",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":TransformsPassIncGen",
        ":VectorOps",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "CommonFolders",
    srcs = [
    ],
    hdrs = [
        "include/mlir/Dialect/CommonFolders.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "LoopsToGPU",
    srcs = ["lib/Conversion/LoopsToGPU/LoopsToGPU.cpp"],
    hdrs = ["include/mlir/Conversion/LoopsToGPU/LoopsToGPU.h"],
    includes = ["include"],
    deps = [
        ":Affine",
        ":AffineToStandardTransforms",
        ":ConversionPassIncGen",
        ":GPUDialect",
        ":GPUTransforms",
        ":IR",
        ":LoopOps",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "LoopsToGPUPass",
    srcs = [
        "lib/Conversion/LoopsToGPU/LoopsToGPUPass.cpp",
        "lib/Conversion/PassDetail.h",
    ],
    hdrs = [
        "include/mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h",
    ],
    includes = ["include"],
    deps = [
        ":Affine",
        ":ConversionPassIncGen",
        ":GPUDialect",
        ":LoopOps",
        ":LoopsToGPU",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "CFGTransforms",
    srcs = [
        "lib/Conversion/LoopToStandard/LoopToStandard.cpp",
        "lib/Conversion/PassDetail.h",
    ],
    hdrs = [
        "include/mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h",
    ],
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":LLVMDialect",
        ":LoopOps",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":Transforms",
    ],
)

cc_library(
    name = "LLVMTransforms",
    srcs = [
        "lib/Conversion/PassDetail.h",
        "lib/Conversion/StandardToLLVM/StandardToLLVM.cpp",
    ],
    hdrs = [
        "include/mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h",
        "include/mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h",
    ],
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":LLVMDialect",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":Transforms",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "CallOpInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "include/mlir/Interfaces/CallInterfaces.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir/Interfaces/CallInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/CallInterfaces.td",
    td_srcs = [
        ":OpBaseTdFiles",
    ],
)

cc_library(
    name = "CallOpInterfaces",
    srcs = [
        "lib/Interfaces/CallInterfaces.cpp",
    ],
    hdrs = [
        "include/mlir/Interfaces/CallInterfaces.h",
    ],
    includes = ["include"],
    deps = [
        ":CallOpInterfacesIncGen",
        ":IR",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "ControlFlowInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "include/mlir/Interfaces/ControlFlowInterfaces.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir/Interfaces/ControlFlowInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/ControlFlowInterfaces.td",
    td_srcs = [
        ":OpBaseTdFiles",
    ],
)

cc_library(
    name = "ControlFlowInterfaces",
    srcs = [
        "lib/Interfaces/ControlFlowInterfaces.cpp",
    ],
    hdrs = [
        "include/mlir/Interfaces/ControlFlowInterfaces.h",
    ],
    includes = ["include"],
    deps = [
        ":ControlFlowInterfacesIncGen",
        ":IR",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "InferTypeOpInterfaceIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "include/mlir/Interfaces/InferTypeOpInterface.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir/Interfaces/InferTypeOpInterface.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/InferTypeOpInterface.td",
    td_srcs = [
        ":OpBaseTdFiles",
    ],
)

cc_library(
    name = "InferTypeOpInterface",
    srcs = [
        "lib/Interfaces/InferTypeOpInterface.cpp",
    ],
    hdrs = [
        "include/mlir/Interfaces/InferTypeOpInterface.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":InferTypeOpInterfaceIncGen",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "SideEffectInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "include/mlir/Interfaces/SideEffectInterfaces.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir/Interfaces/SideEffectInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/SideEffects.td",
    td_srcs = [
        ":OpBaseTdFiles",
    ],
)

cc_library(
    name = "SideEffects",
    srcs = [
        "lib/Interfaces/SideEffects.cpp",
    ],
    hdrs = [
        "include/mlir/Interfaces/SideEffects.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":SideEffectInterfacesIncGen",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "Analysis",
    srcs = glob(
        [
            "lib/Analysis/*.cpp",
            "lib/Analysis/*.h",
        ],
        exclude = [
            "lib/Analysis/Vector*.cpp",
            "lib/Analysis/Vector*.h",
        ],
    ),
    hdrs = glob(
        [
            "include/mlir/Analysis/*.h",
        ],
        exclude = [
            "include/mlir/Analysis/Vector*.h",
        ],
    ),
    includes = ["include"],
    deps = [
        ":Affine",
        ":CallOpInterfaces",
        ":IR",
        ":LoopOps",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "Translation",
    srcs = glob([
        "lib/Translation/*.cpp",
        "lib/Translation/*.h",
    ]),
    hdrs = [
        "include/mlir/Translation.h",
    ],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
        ":Parser",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "LLVMIRModuleTranslation",
    srcs = [
        "lib/Target/LLVMIR/DebugTranslation.cpp",
        "lib/Target/LLVMIR/DebugTranslation.h",
        "lib/Target/LLVMIR/ModuleTranslation.cpp",
    ],
    hdrs = [
        "include/mlir/Target/LLVMIR/ModuleTranslation.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMConversionIncGen",
        ":LLVMDialect",
        ":LLVMIRTransforms",
        ":OpenMPDialect",
        ":Support",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:frontend_open_mp",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:transform_utils",
    ],
)

cc_library(
    name = "TargetLLVMIR",
    srcs = [
        "lib/Target/LLVMIR/ConvertFromLLVMIR.cpp",
        "lib/Target/LLVMIR/ConvertToLLVMIR.cpp",
    ],
    hdrs = [
        "include/mlir/Target/LLVMIR.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMConversionIncGen",
        ":LLVMDialect",
        ":LLVMIRModuleTranslation",
        ":Support",
        ":TargetLLVMAVX512Intr",
        ":Translation",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:ir_reader",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "TargetNVVMIR",
    srcs = [
        "lib/Target/LLVMIR/ConvertToNVVMIR.cpp",
    ],
    hdrs = [
        "include/mlir/Target/NVVMIR.h",
    ],
    includes = ["include"],
    deps = [
        ":GPUDialect",
        ":IR",
        ":LLVMDialect",
        ":LLVMIRModuleTranslation",
        ":NVVMConversionIncGen",
        ":NVVMDialect",
        ":Support",
        ":Translation",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "TargetROCDLIR",
    srcs = [
        "lib/Target/LLVMIR/ConvertToROCDLIR.cpp",
    ],
    hdrs = [
        "include/mlir/Target/ROCDLIR.h",
    ],
    includes = ["include"],
    deps = [
        ":GPUDialect",
        ":IR",
        ":LLVMDialect",
        ":LLVMIRModuleTranslation",
        ":ROCDLConversionIncGen",
        ":ROCDLDialect",
        ":Support",
        ":Translation",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
)

# TODO(zinenko): Update these so that we can simplify mapping to cmake.
cc_library(
    name = "ExecutionEngine",
    srcs = [
        "lib/ExecutionEngine/ExecutionEngine.cpp",
    ],
    hdrs = [
        "include/mlir/ExecutionEngine/ExecutionEngine.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":Support",
        ":TargetLLVMIR",
        ":Translation",
        "@llvm-project//llvm:bit_reader",
        "@llvm-project//llvm:bit_writer",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:execution_engine",
        "@llvm-project//llvm:mc",
        "@llvm-project//llvm:orc_jit",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:target",  # fixdeps: keep
        "@llvm-project//llvm:transform_utils",
        "@llvm-project//llvm:x86_code_gen",  # fixdeps: keep
        "@llvm-project//llvm:x86_disassembler",  # fixdeps: keep
    ],
)

cc_library(
    name = "ExecutionEngineUtils",
    srcs = [
        "lib/ExecutionEngine/OptUtils.cpp",
    ],
    hdrs = [
        "include/mlir/ExecutionEngine/OptUtils.h",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//llvm:analysis",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:ipo",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:target",
    ],
)

# TODO(jpienaar): Update this.
cc_library(
    name = "MlirOptLib",
    srcs = [
        "lib/Support/MlirOptMain.cpp",
    ],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":GPUToNVVMTransforms",
        ":GPUToROCDLTransforms",
        ":GPUToSPIRVTransforms",
        ":GPUTransforms",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":LinalgToLLVM",
        ":LinalgToSPIRV",
        ":LoopOpsTransforms",
        ":NVVMDialect",
        ":Parser",
        ":Pass",
        ":StandardToSPIRVConversions",
        ":StandardToStandard",
        ":Support",
        ":Transforms",
        ":VectorToLLVM",
        ":VectorToLoops",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir/test:TestAffine",
        "@llvm-project//mlir/test:TestDialect",
        "@llvm-project//mlir/test:TestIR",
        "@llvm-project//mlir/test:TestPass",
        "@llvm-project//mlir/test:TestSPIRV",
        "@llvm-project//mlir/test:TestTransforms",
    ],
)

cc_library(
    name = "AllTranslations",
    hdrs = ["include/mlir/InitAllTranslations.h"],
    deps = [
        ":SPIRVTranslateRegistration",
        ":TargetLLVMIR",
        ":TargetNVVMIR",
        ":TargetROCDLIR",
    ],
)

cc_library(
    name = "MlirTranslateMain",
    srcs = ["tools/mlir-translate/mlir-translate.cpp"],
    deps = [
        ":AllPassesAndDialectsNoRegistration",
        ":AllTranslations",
        ":IR",
        ":Parser",
        ":Support",
        ":Translation",
        "@llvm-project//llvm:support",
    ],
)

cc_binary(
    name = "mlir-translate",
    deps = [
        ":MlirTranslateMain",
    ],
)

cc_library(
    name = "AllPassesAndDialectsNoRegistration",
    hdrs = [
        "include/mlir/InitAllDialects.h",
        "include/mlir/InitAllPasses.h",
    ],
    defines = ["MLIR_CUDA_CONVERSIONS_ENABLED"],
    deps = [
        ":AVX512",
        ":AVX512ToLLVM",
        ":Affine",
        ":AffinePassIncGen",
        ":AffineTransforms",
        ":CFGTransforms",
        ":ConversionPassIncGen",
        ":GPUDialect",
        ":GPUPassIncGen",
        ":GPUToCUDATransforms",
        ":GPUToNVVMTransforms",
        ":GPUToROCDLTransforms",
        ":GPUToSPIRVTransforms",
        ":GPUToVulkanTransforms",
        ":GPUTransforms",
        ":IR",
        ":LLVMAVX512",
        ":LLVMDialect",
        ":LLVMIRTransforms",
        ":LLVMPassIncGen",
        ":LLVMTransforms",
        ":LinalgOps",
        ":LinalgPassIncGen",
        ":LinalgToLLVM",
        ":LinalgToSPIRV",
        ":LinalgTransforms",
        ":LoopOps",
        ":LoopOpsTransforms",
        ":LoopPassIncGen",
        ":LoopsToGPUPass",
        ":NVVMDialect",
        ":OpenMPDialect",
        ":QuantOps",
        ":QuantPassIncGen",
        ":ROCDLDialect",
        ":SDBM",
        ":SPIRVDialect",
        ":SPIRVLowering",
        ":SPIRVPassIncGen",
        ":Shape",
        ":StandardOps",
        ":StandardToSPIRVConversions",
        ":StandardToStandard",
        ":Transforms",
        ":TransformsPassIncGen",
        ":VectorOps",
        ":VectorToLLVM",
    ],
)

cc_library(
    name = "AllPassesAndDialects",
    # srcs = ["@com_intel_plaidml//vendor/mlir:mlir-auto-init.cpp"],
    deps = [
        ":AllPassesAndDialectsNoRegistration",
    ],
    alwayslink = 1,
)

# TODO(jpienaar): This library should be removed.
cc_library(
    name = "MlirOptMain",
    srcs = [
        "tools/mlir-opt/mlir-opt.cpp",
    ],
    deps = [
        ":AllPassesAndDialectsNoRegistration",
        ":Analysis",
        ":IR",
        ":MlirOptLib",
        ":Pass",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_binary(
    name = "mlir-opt",
    deps = [
        ":Analysis",
        ":IR",
        ":LoopsToGPUPass",
        ":MlirOptLib",
        ":MlirOptMain",
        ":OpenMPDialect",
        ":QuantOps",
        ":Transforms",
        "@llvm-project//llvm:all_targets",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir/test:TestAffine",
        "@llvm-project//mlir/test:TestDialect",
        "@llvm-project//mlir/test:TestIR",
        "@llvm-project//mlir/test:TestPass",
        "@llvm-project//mlir/test:TestSPIRV",
        "@llvm-project//mlir/test:TestTransforms",
    ],
)

cc_library(
    name = "MlirJitRunner",
    srcs = ["lib/Support/JitRunner.cpp"],
    hdrs = ["include/mlir/Support/JitRunner.h"],
    includes = ["include"],
    deps = [
        ":AllPassesAndDialectsNoRegistration",
        ":CFGTransforms",
        ":ExecutionEngine",
        ":ExecutionEngineUtils",
        ":IR",
        ":LLVMDialect",
        ":Parser",
        ":Pass",
        ":Support",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:orc_jit",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "mlir_c_runner_utils",
    srcs = [
        "lib/ExecutionEngine/CRunnerUtils.cpp",
    ],
    hdrs = [
        "include/mlir/ExecutionEngine/CRunnerUtils.h",
    ],
    includes = ["include"],
)

cc_library(
    name = "mlir_runner_utils",
    srcs = [
        "lib/ExecutionEngine/RunnerUtils.cpp",
    ],
    hdrs = [
        "include/mlir/ExecutionEngine/RunnerUtils.h",
    ],
    includes = ["include"],
    deps = [
        ":mlir_c_runner_utils",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "mlir-cpu-runner",
    srcs = ["tools/mlir-cpu-runner/mlir-cpu-runner.cpp"],
    linkopts = ["-ldl"],
    deps = [
        ":AllPassesAndDialectsNoRegistration",
        ":ExecutionEngineUtils",
        ":MlirJitRunner",
        "@llvm-project//llvm:support",
    ],
)

# cc_binary(
#     name = "tools/libcuda-runtime-wrappers.so",
#     srcs = ["tools/mlir-cuda-runner/cuda-runtime-wrappers.cpp"],
#     linkshared = True,
#     deps = [
#         "//third_party/gpus/cuda:cuda_headers",
#         "//third_party/gpus/cuda:cuda_runtime",
#         "//third_party/gpus/cuda:libcuda",
#         "@llvm-project//llvm:support",
#     ],
# )

# cc_binary(
#     name = "mlir-cuda-runner",
#     srcs = ["tools/mlir-cuda-runner/mlir-cuda-runner.cpp"],
#     data = [
#         ":tools/libcuda-runtime-wrappers.so",
#         "@llvm-project//mlir/test/mlir-cpu-runner:libmlir_runner_utils.so",
#     ],
#     deps = [
#         ":AllPassesAndDialectsNoRegistration",
#         ":ExecutionEngineUtils",
#         ":GPUDialect",
#         ":GPUToNVVMTransforms",
#         ":GPUToROCDLTransforms",
#         ":GPUTransforms",
#         ":IR",
#         ":LLVMDialect",
#         ":LLVMTransforms",
#         ":MlirJitRunner",
#         ":NVVMDialect",
#         ":Pass",
#         ":Transforms",
#         "//devtools/build/runtime:get_runfiles_dir",
#         "//third_party/gpus/cuda:cuda_headers",
#         "//third_party/gpus/cuda:cuda_runtime",
#         "//third_party/gpus/cuda:libcuda",
#         "@llvm-project//llvm:support",
#     ],
# )

cc_library(
    name = "TableGen",
    srcs = glob(["lib/TableGen/*.cpp"]),
    hdrs = glob(["include/mlir/TableGen/*.h"]),
    includes = ["include"],
    deps = [
        ":Support",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:tablegen",
    ],
)

cc_library(
    name = "MlirTableGenMain",
    srcs = [
        "tools/mlir-tblgen/mlir-tblgen.cpp",
    ],
    includes = ["include"],
    deps = [
        ":Support",
        ":TableGen",
        "@llvm-project//llvm:config",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:tablegen",
    ],
)

cc_binary(
    name = "mlir-tblgen",
    srcs = glob([
        "tools/mlir-tblgen/*.h",
        "tools/mlir-tblgen/*.cpp",
    ]),
    linkopts = LINKOPTS,
    deps = [
        ":MlirTableGenMain",
        ":Support",
        ":TableGen",
        "@llvm-project//llvm:config",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:tablegen",
    ],
)

cc_binary(
    name = "mlir-linalg-ods-gen",
    srcs = glob([
        "tools/mlir-linalg-ods-gen/mlir-linalg-ods-gen.cpp",
    ]),
    deps = [
        ":IR",
        ":Support",
        "@llvm-project//llvm:config",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:tablegen",
    ],
)

## OpenMP dialect
gentbl(
    name = "OpenMPOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/OpenMP/OpenMPOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/OpenMP/OpenMPOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls",
            "include/mlir/Dialect/OpenMP/OpenMPOpsDialect.h.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/Dialects/OpenMP/OpenMPOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/OpenMP/OpenMPOps.td",
    td_srcs = [
        ":OpBaseTdFiles",
    ],
)

cc_library(
    name = "OpenMPDialect",
    srcs = glob(
        [
            "lib/Dialect/OpenMP/IR/*.cpp",
            "lib/Dialect/OpenMP/IR/*.h",
        ],
    ),
    hdrs = glob([
        "include/mlir/Dialect/OpenMP/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":IR",
        ":OpenMPOpsIncGen",
        "@llvm-project//llvm:support",
    ],
)

filegroup(
    name = "QuantizationOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Quant/QuantOps.td",
        "include/mlir/Dialect/Quant/QuantOpsBase.td",
        "include/mlir/Interfaces/SideEffects.td",
        ":OpBaseTdFiles",
    ],
)

## QuantOps dialect
gentbl(
    name = "QuantOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/Quant/QuantOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/Quant/QuantOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls",
            "include/mlir/Dialect/Quant/QuantOpsDialect.h.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/Dialects/QuantOps/QuantOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Quant/QuantOps.td",
    td_srcs = [
        ":QuantizationOpsTdFiles",
    ],
)

gentbl(
    name = "QuantPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-pass-decls",
            "include/mlir/Dialect/Quant/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Quant/Passes.td",
    td_srcs = [
        ":PassBaseTdFiles",
    ],
)

cc_library(
    name = "QuantOps",
    srcs = [
        "lib/Dialect/Quant/IR/QuantOps.cpp",
        "lib/Dialect/Quant/IR/QuantTypes.cpp",
        "lib/Dialect/Quant/IR/TypeDetail.h",
        "lib/Dialect/Quant/IR/TypeParser.cpp",
        "lib/Dialect/Quant/Transforms/ConvertConst.cpp",
        "lib/Dialect/Quant/Transforms/ConvertSimQuant.cpp",
        "lib/Dialect/Quant/Transforms/PassDetail.h",
        "lib/Dialect/Quant/Utils/FakeQuantSupport.cpp",
        "lib/Dialect/Quant/Utils/QuantizeUtils.cpp",
        "lib/Dialect/Quant/Utils/UniformSupport.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/Quant/FakeQuantSupport.h",
        "include/mlir/Dialect/Quant/Passes.h",
        "include/mlir/Dialect/Quant/QuantOps.h",
        "include/mlir/Dialect/Quant/QuantTypes.h",
        "include/mlir/Dialect/Quant/QuantizeUtils.h",
        "include/mlir/Dialect/Quant/UniformSupport.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":Pass",
        ":QuantOpsIncGen",
        ":QuantPassIncGen",
        ":SideEffects",
        ":StandardOps",
        "@llvm-project//llvm:support",
    ],
)

filegroup(
    name = "LinalgOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Linalg/IR/LinalgBase.td",
        "include/mlir/Dialect/Linalg/IR/LinalgOps.td",
        ":AffineOpsTdFiles",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "LinalgOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/Linalg/IR/LinalgOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls -dialect=linalg",
            "include/mlir/Dialect/Linalg/IR/LinalgOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/IR/LinalgOps.td",
    td_srcs = [
        ":LinalgOpsTdFiles",
    ],
)

filegroup(
    name = "LinalgStructuredOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td",
        "include/mlir/Dialect/Linalg/IR/LinalgStructuredOpsInterface.td",
        ":AffineOpsTdFiles",
        ":LinalgOpsTdFiles",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "LinalgStructuredOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc",
        ),
        (
            "-gen-op-interface-decls",
            "include/mlir/Dialect/Linalg/IR/LinalgStructuredOpsInterfaces.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir/Dialect/Linalg/IR/LinalgStructuredOpsInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td",
    td_srcs = [
        ":LinalgStructuredOpsTdFiles",
    ],
)

filegroup(
    name = "LinalgDocTdFiles",
    srcs = [
        "include/mlir/Dialect/Linalg/IR/LinalgDoc.td",
        ":LinalgOpsTdFiles",
        ":LinalgStructuredOpsTdFiles",
    ],
)

gentbl(
    name = "LinalgDocIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-doc",
            "g3doc/Dialects/Linalg/LinalgOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/IR/LinalgDoc.td",
    td_srcs = [
        ":LinalgDocTdFiles",
    ],
)

filegroup(
    name = "LinalgTransformPatternsTdFiles",
    srcs = [
        "include/mlir/Dialect/Linalg/Transforms/LinalgTransformPatterns.td",
        ":AffineOpsTdFiles",
        ":LinalgOpsTdFiles",
        ":LinalgStructuredOpsTdFiles",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "LinalgTransformPatternsIncGen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "include/mlir/Dialect/Linalg/Transforms/LinalgTransformPatterns.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/Transforms/LinalgTransformPatterns.td",
    td_srcs = [
        ":LinalgTransformPatternsTdFiles",
    ],
)

cc_library(
    name = "LinalgToLLVM",
    srcs = glob([
        "lib/Conversion/LinalgToLLVM/*.cpp",
        "lib/Conversion/LinalgToLLVM/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/LinalgToLLVM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":AffineToStandardTransforms",
        ":Analysis",
        ":CFGTransforms",
        ":ConversionPassIncGen",
        ":EDSC",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":LinalgOps",
        ":LinalgTransforms",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":VectorToLLVM",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "LinalgToSPIRV",
    srcs = glob([
        "lib/Conversion/LinalgToSPIRV/*.cpp",
        "lib/Conversion/LinalgToSPIRV/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/LinalgToSPIRV/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":DialectUtils",
        ":IR",
        ":LinalgOps",
        ":LinalgTransforms",
        ":Pass",
        ":SPIRVDialect",
        ":SPIRVLowering",
        ":StandardOps",
    ],
)

cc_library(
    name = "LinalgOps",
    srcs = [
        "lib/Dialect/Linalg/IR/LinalgOps.cpp",
        "lib/Dialect/Linalg/IR/LinalgTypes.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/Linalg/IR/LinalgOps.h",
        "include/mlir/Dialect/Linalg/IR/LinalgTraits.h",
        "include/mlir/Dialect/Linalg/IR/LinalgTypes.h",
    ],
    includes = ["include"],
    deps = [
        ":DialectUtils",
        ":IR",
        ":LinalgOpsIncGen",
        ":LinalgStructuredOpsIncGen",
        ":Parser",
        ":SideEffects",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "LinalgPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-pass-decls",
            "include/mlir/Dialect/Linalg/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/Passes.td",
    td_srcs = [
        ":PassBaseTdFiles",
    ],
)

cc_library(
    name = "LinalgTransforms",
    srcs = glob([
        "lib/Dialect/Linalg/Transforms/*.cpp",
        "lib/Dialect/Linalg/Transforms/*.h",
    ]) + [
        "lib/Dialect/Linalg/Analysis/DependenceAnalysis.cpp",
        "lib/Dialect/Linalg/EDSC/Builders.cpp",
        "lib/Dialect/Linalg/Utils/Utils.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h",
        "include/mlir/Dialect/Linalg/EDSC/Builders.h",
        "include/mlir/Dialect/Linalg/EDSC/Intrinsics.h",
        "include/mlir/Dialect/Linalg/Passes.h",
        "include/mlir/Dialect/Linalg/Transforms/LinalgTransforms.h",
        "include/mlir/Dialect/Linalg/Utils/Utils.h",
    ],
    includes = ["include"],
    deps = [
        ":Affine",
        ":AffineToStandardTransforms",
        ":Analysis",
        ":CFGTransforms",
        ":DialectUtils",
        ":EDSC",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":LinalgOps",
        ":LinalgPassIncGen",
        ":LinalgStructuredOpsIncGen",
        ":LoopOps",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":Transforms",
        ":TransformsPassIncGen",
        ":VectorOps",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
)

filegroup(
    name = "VectorOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Vector/VectorOps.td",
        ":AffineOpsTdFiles",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "VectorOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/Vector/VectorOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/Vector/VectorOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls -dialect=vector",
            "include/mlir/Dialect/Vector/VectorOpsDialect.h.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/Dialects/Vector/VectorOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Vector/VectorOps.td",
    td_srcs = [
        ":VectorOpsTdFiles",
    ],
)

filegroup(
    name = "VectorTransformPatternsTdFiles",
    srcs = [
        "include/mlir/Dialect/Vector/VectorTransformPatterns.td",
        ":AffineOpsTdFiles",
        ":LinalgOpsTdFiles",
        ":LinalgStructuredOpsTdFiles",
        ":OpBaseTdFiles",
        ":StdOpsTdFiles",
        ":VectorOpsTdFiles",
    ],
)

gentbl(
    name = "VectorTransformPatternsIncGen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "include/mlir/Dialect/Vector/VectorTransformPatterns.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Vector/VectorTransformPatterns.td",
    td_srcs = [
        ":VectorTransformPatternsTdFiles",
    ],
)

cc_library(
    name = "VectorToLLVM",
    srcs = glob([
        "lib/Conversion/VectorToLLVM/*.cpp",
        "lib/Conversion/VectorToLLVM/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/VectorToLLVM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":DialectUtils",
        ":EDSC",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":VectorOps",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "VectorToLoops",
    srcs = glob([
        "lib/Conversion/VectorToLoops/*.cpp",
        "lib/Conversion/VectorToLoops/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Conversion/VectorToLoops/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":Affine",
        ":ConversionPassIncGen",
        ":EDSC",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":LoopOps",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":VectorOps",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
)

# To reference all tablegen files here when checking for updates to them.
filegroup(
    name = "TdFiles",
    srcs = glob(["**/*.td"]),
)

exports_files(
    [
        "include/mlir/Interfaces/CallInterfaces.h",
        "include/mlir/Interfaces/CallInterfaces.td",
        "include/mlir/Interfaces/ControlFlowInterfaces.h",
        "include/mlir/Interfaces/ControlFlowInterfaces.td",
        "include/mlir/Interfaces/SideEffects.td",
        "include/mlir/Dialect/LLVMIR/LLVMOpBase.td",
        "include/mlir/Dialect/StandardOps/IR/Ops.td",
        "include/mlir/IR/OpAsmInterface.td",
        "include/mlir/IR/OpBase.td",
        "include/mlir/Transforms/InliningUtils.h",
    ],
    visibility = [":friends"],
)

exports_files(
    [
        "include/mlir/Interfaces/InferTypeOpInterface.td",
        "include/mlir/Interfaces/LoopLikeInterface.td",
    ],
    visibility = [":friends"],
)
