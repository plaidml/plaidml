# (PlaidML)
load("@com_intel_plaidml//vendor/mlir:tblgen.bzl", "gentbl")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(
    default_visibility = [":test_friends"],
    licenses = ["notice"],
)

# Please only depend on this from MLIR tests.
package_group(
    name = "test_friends",
    packages = ["//..."],
)

cc_library(
    name = "IRProducingAPITest",
    hdrs = ["APITest.h"],
    includes = ["."],
)

filegroup(
    name = "TestOpTdFiles",
    srcs = [
        "lib/Dialect/Test/TestInterfaces.td",
        "lib/Dialect/Test/TestOps.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@llvm-project//mlir:include/mlir/IR/OpAsmInterface.td",
        "@llvm-project//mlir:include/mlir/IR/RegionKindInterface.td",
        "@llvm-project//mlir:include/mlir/IR/SymbolInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/CallInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/ControlFlowInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/CopyOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
    ],
)

gentbl(
    name = "TestOpsIncGen",
    strip_include_prefix = "lib/Dialect/Test",
    tbl_outs = [
        (
            "-gen-op-decls",
            "lib/Dialect/Test/TestOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "lib/Dialect/Test/TestOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls",
            "lib/Dialect/Test/TestOpsDialect.h.inc",
        ),
        (
            "-gen-enum-decls",
            "lib/Dialect/Test/TestOpEnums.h.inc",
        ),
        (
            "-gen-enum-defs",
            "lib/Dialect/Test/TestOpEnums.cpp.inc",
        ),
        (
            "-gen-struct-attr-decls",
            "lib/Dialect/Test/TestOpStructs.h.inc",
        ),
        (
            "-gen-struct-attr-defs",
            "lib/Dialect/Test/TestOpStructs.cpp.inc",
        ),
        (
            "-gen-rewriters",
            "lib/Dialect/Test/TestPatterns.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/Test/TestOps.td",
    td_srcs = [
        ":TestOpTdFiles",
    ],
    test = True,
)

gentbl(
    name = "TestInterfacesIncGen",
    strip_include_prefix = "lib/Dialect/Test",
    tbl_outs = [
        (
            "-gen-type-interface-decls",
            "lib/Dialect/Test/TestTypeInterfaces.h.inc",
        ),
        (
            "-gen-type-interface-defs",
            "lib/Dialect/Test/TestTypeInterfaces.cpp.inc",
        ),
        (
            "-gen-op-interface-decls",
            "lib/Dialect/Test/TestOpInterfaces.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "lib/Dialect/Test/TestOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/Test/TestInterfaces.td",
    td_srcs = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectBaseTdFiles",
    ],
    test = True,
)

gentbl(
    name = "TestTypeDefsIncGen",
    strip_include_prefix = "lib/Dialect/Test",
    tbl_outs = [
        (
            "-gen-typedef-decls",
            "lib/Dialect/Test/TestTypeDefs.h.inc",
        ),
        (
            "-gen-typedef-defs",
            "lib/Dialect/Test/TestTypeDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/Test/TestTypeDefs.td",
    td_srcs = [
        ":TestOpTdFiles",
    ],
    test = True,
)

cc_library(
    name = "TestDialect",
    srcs = [
        "lib/Dialect/Test/TestDialect.cpp",
        "lib/Dialect/Test/TestInterfaces.cpp",
        "lib/Dialect/Test/TestPatterns.cpp",
        "lib/Dialect/Test/TestTraits.cpp",
        "lib/Dialect/Test/TestTypes.cpp",
    ],
    hdrs = [
        "lib/Dialect/Test/TestDialect.h",
        "lib/Dialect/Test/TestInterfaces.h",
        "lib/Dialect/Test/TestTypes.h",
    ],
    includes = [
        "lib/Dialect/Test",
    ],
    deps = [
        ":TestInterfacesIncGen",
        ":TestOpsIncGen",
        ":TestTypeDefsIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:CopyOpInterface",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TestIR",
    srcs = [
        "lib/IR/TestFunc.cpp",
        "lib/IR/TestInterfaces.cpp",
        "lib/IR/TestMatchers.cpp",
        "lib/IR/TestPrintDefUse.cpp",
        "lib/IR/TestPrintNesting.cpp",
        "lib/IR/TestSideEffects.cpp",
        "lib/IR/TestSlicing.cpp",
        "lib/IR/TestSymbolUses.cpp",
        "lib/IR/TestTypes.cpp",
    ],
    deps = [
        ":TestDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "TestPass",
    srcs = [
        "lib/Pass/TestPassManager.cpp",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "TestRewrite",
    srcs = [
        "lib/Rewrite/TestPDLByteCode.cpp",
    ],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "TestReducer",
    srcs = [
        "lib/Reducer/MLIRTestReducer.cpp",
    ],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "TestTransforms",
    srcs = glob(["lib/Transforms/*.cpp"]),
    defines = ["MLIR_CUDA_CONVERSIONS_ENABLED"],
    includes = ["lib/Dialect/Test"],
    deps = [
        ":TestDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TargetNVVMIR",
        "@llvm-project//mlir:TargetROCDLIR",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:VectorToSCF",
    ],
)

cc_library(
    name = "TestAffine",
    srcs = glob([
        "lib/Dialect/Affine/*.cpp",
    ]),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
    ],
)

cc_library(
    name = "TestShapeDialect",
    srcs = [
        "lib/Dialect/Shape/TestShapeFunctions.cpp",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
    ],
)

cc_library(
    name = "TestSPIRV",
    srcs = glob([
        "lib/Dialect/SPIRV/*.cpp",
    ]),
    deps = [
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVConversion",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVModuleCombiner",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TestTypeDialect",
    srcs = glob([
        "lib/Dialect/LLVMIR/*.cpp",
    ]),
    deps = [
        ":TestDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
    ],
)

cc_library(
    name = "TestTosaDialect",
    srcs = glob([
        "lib/Dialect/Tosa/*.cpp",
    ]),
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:Transforms",
    ],
)
