load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("@com_intel_plaidml//vendor/llvm:build_defs.bzl", "tblgen")

OUT_DIR = "include/llvm/"

CFG_FILES = [
    OUT_DIR + "Config/AsmParsers.def",
    OUT_DIR + "Config/AsmPrinters.def",
    OUT_DIR + "Config/config.h",
    OUT_DIR + "Config/abi-breaking.h",
    OUT_DIR + "Config/Disassemblers.def",
    OUT_DIR + "Config/llvm-config.h",
    OUT_DIR + "Config/Targets.def",
]

PLATFORM_COPTS = select({
    "@toolchain//:macos_x86_64": [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
        "-w",
    ],
    "@toolchain//:windows_x86_64": [
        "/w",
        "/wd4244",
        "/wd4267",
    ],
    "//conditions:default": [
        "-fPIC",
        "-w",
    ],
})

# Local GenRule to configure llvm
genrule(
    name = "configure",
    srcs = ["CMakeLists.txt"],
    outs = CFG_FILES,
    cmd = select({
        "@toolchain//:macos_x86_64": """
cmake -B$(@D) -H$$(dirname $(location //:CMakeLists.txt)) \
    -DPYTHON_EXECUTABLE=$$(which python3) \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DHAVE_LIBEDIT=0 \
    -DHAVE_FUTIMENS=0 \
    -DHAVE_VALGRIND_VALGRIND_H=0 \
    -DLLVM_TARGETS_TO_BUILD="X86"
""",
        "@toolchain//:windows_x86_64": """
$${CONDA_PREFIX}/Library/bin/cmake -Thost=x64 -B$(@D) -H$$(dirname $(location //:CMakeLists.txt)) \
    -DPYTHON_EXECUTABLE=$$(which python) \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DHAVE_VALGRIND_VALGRIND_H=0 \
    -DLLVM_TARGETS_TO_BUILD="X86"
""",
        "//conditions:default": """
cmake -B$(@D) -H$$(dirname $(location //:CMakeLists.txt)) \
    -DPYTHON_EXECUTABLE=$$(which python3) \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DHAVE_VALGRIND_VALGRIND_H=0 \
    -DLLVM_TARGETS_TO_BUILD="X86"
""",
    }),
)

cc_library(
    name = "Demangle",
    srcs = glob([
        "lib/Demangle/**/*.cpp",
        "lib/Demangle/**/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Demangle/**/*.h",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "Support",
    srcs = glob([
        "lib/Support/**/*.cpp",
        "lib/Support/**/*.c",
        "lib/Support/**/*.h",
        "lib/Support/**/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/ADT/**/*.h",
        "include/llvm/Support/**/*.h",
    ]) + CFG_FILES,
    copts = PLATFORM_COPTS,
    includes = ["include"],
    linkopts = select({
        "@toolchain//:windows_x86_64": [],
        "@toolchain//:macos_x86_64": [],
        "//conditions:default": [
            "-pthread",
            "-ldl",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":Demangle",
        "@zlib",
    ],
)

cc_library(
    name = "Core",
    srcs = glob([
        "lib/IR/**/*.cpp",
        "lib/IR/**/*.h",
        "lib/IR/**/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/IR/**/*.h",
    ]) + [
        ":gen-attrs",
        ":gen-attrs-compat",
        ":gen-intrinsic-enums",
        ":gen-intrinsic-impl",
    ],
    copts = PLATFORM_COPTS + select({
        "@toolchain//:windows_x86_64": [
            "/I$(GENDIR)/external/llvm/lib/IR",
        ],
        "//conditions:default": [
            "-iquote",
            "$(GENDIR)/external/llvm/lib/IR",
        ],
    }),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":BinaryFormat",
        ":Remarks",
        ":Support",
    ],
)

cc_library(
    name = "TableGen",
    srcs = glob([
        "lib/TableGen/**/*.cpp",
        "lib/TableGen/**/*.h",
    ]),
    hdrs = glob([
        "include/llvm/TableGen/**/*.h",
    ]),
    copts = PLATFORM_COPTS,
    linkopts = select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": ["-lm"],
    }),
    visibility = ["//visibility:public"],
    deps = [":Support"],
)

cc_library(
    name = "DebugInfoCodeView",
    srcs = glob([
        "lib/DebugInfo/CodeView/*.cpp",
        "lib/DebugInfo/CodeView/*.h",
    ]),
    hdrs = glob([
        "include/llvm/DebugInfo/CodeView/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":DebugInfoMSF",
        ":Support",
    ],
)

cc_library(
    name = "DebugInfoDWARF",
    srcs = glob([
        "lib/DebugInfo/DWARF/*.cpp",
        "lib/DebugInfo/DWARF/*.h",
    ]),
    hdrs = glob([
        "include/llvm/DebugInfo/DWARF/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":BinaryFormat",
        ":MC",
        ":Object",
        ":Support",
    ],
)

cc_library(
    name = "DebugInfoMSF",
    srcs = glob([
        "lib/DebugInfo/MSF/*.cpp",
        "lib/DebugInfo/MSF/*.h",
    ]),
    hdrs = glob([
        "include/llvm/DebugInfo/MSF/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Support",
    ],
)

cc_library(
    name = "ExecutionEngine",
    srcs = glob([
        "lib/ExecutionEngine/*.cpp",
        "lib/ExecutionEngine/*.h",
    ]),
    hdrs = glob([
        "include/llvm/ExecutionEngine/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Core",
        ":MC",
        ":Object",
        ":RuntimeDyld",
        ":Support",
        ":Target",
    ],
)

cc_library(
    name = "Remarks",
    srcs = glob([
        "lib/Remarks/*.cpp",
        "lib/Remarks/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Remarks/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [":Support"],
)

cc_library(
    name = "RuntimeDyld",
    srcs = glob([
        "lib/ExecutionEngine/RuntimeDyld/**/*.cpp",
        "lib/ExecutionEngine/RuntimeDyld/**/*.h",
    ]),
    hdrs = glob([
        "include/llvm/ExecutionEngine/RuntimeDyld/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [":Core"],
)

cc_library(
    name = "Object",
    srcs = glob([
        "lib/Object/*.cpp",
        "lib/Object/*.h",
    ]) + [
        ":llvm_vcsrevision_h",
    ],
    hdrs = glob([
        "include/llvm/Object/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":BinaryFormat",
        ":BitReader",
        ":Core",
        ":MC",
        ":MCParser",
        ":Support",
    ],
)

cc_library(
    name = "MCJIT",
    srcs = glob([
        "lib/ExecutionEngine/MCJIT/*.cpp",
        "lib/ExecutionEngine/MCJIT/*.h",
    ]),
    hdrs = glob([
        "include/llvm/ExecutionEngine/MCJIT/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Core",
        ":ExecutionEngine",
        ":Object",
        ":RuntimeDyld",
        ":Support",
        ":Target",
    ],
)

cc_library(
    name = "OrcJIT",
    srcs = glob([
        "lib/ExecutionEngine/Orc/*.cpp",
        "lib/ExecutionEngine/Orc/*.h",
    ]),
    hdrs = glob([
        "include/llvm/ExecutionEngine/Orc/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Core",
        ":ExecutionEngine",
        ":JITLink",
        ":MC",
        ":Object",
        ":RuntimeDyld",
        ":Support",
        ":Target",
        ":TransformUtils",
    ],
)

cc_library(
    name = "JITLink",
    srcs = glob([
        "lib/ExecutionEngine/JITLink/*.cpp",
        "lib/ExecutionEngine/JITLink/*.h",
    ]),
    hdrs = glob([
        "include/llvm/ExecutionEngine/JITLink/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":BinaryFormat",
        ":Object",
        ":Support",
    ],
)

cc_library(
    name = "AsmPrinter",
    srcs = glob([
        "lib/CodeGen/AsmPrinter/*.cpp",
        "lib/CodeGen/AsmPrinter/*.h",
    ]),
    hdrs = glob([
        "include/llvm/CodeGen/AsmPrinter/*.h",
        "lib/CodeGen/AsmPrinter/*.def",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Analysis",
        ":BinaryFormat",
        ":CodeGen",
        ":Core",
        ":DebugInfoCodeView",
        ":DebugInfoDWARF",
        ":DebugInfoMSF",
        ":MC",
        ":MCParser",
        ":Remarks",
        ":Support",
        ":Target",
    ],
)

cc_library(
    name = "CodeGen",
    srcs = glob([
        "lib/CodeGen/*.cpp",
        "lib/CodeGen/*.h",
    ]),
    hdrs = glob([
        "include/llvm/CodeGen/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Analysis",
        ":BitReader",
        ":BitWriter",
        ":Core",
        ":MC",
        ":ProfileData",
        ":Scalar",
        ":Support",
        ":Target",
        ":TransformUtils",
    ],
)

cc_library(
    name = "GlobalISel",
    srcs = glob([
        "lib/CodeGen/GlobalISel/*.cpp",
        "lib/CodeGen/GlobalISel/*.h",
    ]),
    hdrs = glob([
        "include/llvm/CodeGen/GlobalISel/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Analysis",
        ":CodeGen",
        ":Core",
        ":MC",
        ":Support",
        ":Target",
        ":TransformUtils",
    ],
)

cc_library(
    name = "SelectionDAG",
    srcs = glob([
        "lib/CodeGen/SelectionDAG/*.cpp",
        "lib/CodeGen/SelectionDAG/*.h",
    ]),
    hdrs = glob([
        "include/llvm/CodeGen/SelectionDAG/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Analysis",
        ":CodeGen",
        ":Core",
        ":MC",
        ":Support",
        ":Target",
        ":TransformUtils",
    ],
)

cc_library(
    name = "MC",
    srcs = glob([
        "lib/MC/*.cpp",
        "lib/MC/*.h",
    ]),
    hdrs = glob([
        "include/llvm/MC/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":BinaryFormat",
        ":DebugInfoCodeView",
        ":Support",
    ],
)

cc_library(
    name = "MCDisassembler",
    srcs = glob([
        "lib/MC/MCDisassembler/*.cpp",
        "lib/MC/MCDisassembler/*.h",
    ]),
    hdrs = glob([
        "include/llvm/MC/MCDisassembler/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":MC",
        ":Support",
    ],
)

cc_library(
    name = "MCParser",
    srcs = glob([
        "lib/MC/MCParser/*.cpp",
        "lib/MC/MCParser/*.h",
    ]),
    hdrs = glob([
        "include/llvm/MC/MCParser/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":MC",
        ":Support",
    ],
)

cc_library(
    name = "BinaryFormat",
    srcs = glob([
        "lib/BinaryFormat/*.cpp",
        "lib/BinaryFormat/*.h",
    ]),
    hdrs = glob([
        "include/llvm/BinaryFormat/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Support",
    ],
)

cc_library(
    name = "BitReader",
    srcs = glob([
        "lib/Bitcode/Reader/*.cpp",
        "lib/Bitcode/Reader/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Bitcode/Reader/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Core",
        ":Support",
    ],
)

cc_library(
    name = "BitWriter",
    srcs = glob([
        "lib/Bitcode/Writer/*.cpp",
        "lib/Bitcode/Writer/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Bitcode/Writer/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Analysis",
        ":Core",
        ":MC",
        ":Object",
        ":Support",
    ],
)

cc_library(
    name = "AsmParser",
    srcs = glob([
        "lib/AsmParser/*.cpp",
        "lib/AsmParser/*.h",
    ]),
    hdrs = glob([
        "include/llvm/AsmParser/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":BinaryFormat",
        ":Core",
        ":Support",
    ],
)

cc_library(
    name = "TransformUtils",
    srcs = glob([
        "lib/Transforms/Utils/*.cpp",
        "lib/Transforms/Utils/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Transforms/Utils/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Analysis",
        ":Core",
        ":Support",
    ],
)

cc_library(
    name = "AggressiveInstCombine",
    srcs = glob([
        "lib/Transforms/AggressiveInstCombine/*.cpp",
        "lib/Transforms/AggressiveInstCombine/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Transforms/AggressiveInstCombine/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Analysis",
        ":Core",
        ":Support",
        ":TransformUtils",
    ],
)

cc_library(
    name = "IRReader",
    srcs = glob([
        "lib/IRReader/*.cpp",
        "lib/IRReader/*.h",
    ]),
    hdrs = glob([
        "include/llvm/IRReader/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":AsmParser",
        ":BitReader",
        ":Core",
        ":Support",
    ],
)

cc_library(
    name = "InstCombine",
    srcs = glob([
        "lib/Transforms/InstCombine/*.cpp",
        "lib/Transforms/InstCombine/*.h",
    ]) + [
        ":gen-inst-combine-tables",
    ],
    hdrs = glob([
        "include/llvm/Transforms/InstCombine/*.h",
    ]),
    copts = PLATFORM_COPTS + select({
        "@toolchain//:windows_x86_64": [
            "/I$(GENDIR)/external/llvm/lib/Transforms/InstCombine",
        ],
        "//conditions:default": [
            "-iquote",
            "$(GENDIR)/external/llvm/lib/Transforms/InstCombine",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":Analysis",
        ":Core",
        ":Support",
        ":TransformUtils",
    ],
)

cc_library(
    name = "Instrumentation",
    srcs = glob([
        "lib/Transforms/Instrumentation/*.cpp",
        "lib/Transforms/Instrumentation/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Transforms/Instrumentation/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Analysis",
        ":Core",
        ":MC",
        ":ProfileData",
        ":Support",
        ":TransformUtils",
    ],
)

cc_library(
    name = "Linker",
    srcs = glob([
        "lib/Linker/*.cpp",
        "lib/Linker/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Linker/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Core",
        ":Support",
        ":TransformUtils",
    ],
)

cc_library(
    name = "Scalar",
    srcs = glob([
        "lib/Transforms/Scalar/*.cpp",
        "lib/Transforms/Scalar/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Transforms/Scalar/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":AggressiveInstCombine",
        ":Analysis",
        ":Core",
        ":InstCombine",
        ":Support",
        ":TransformUtils",
    ],
)

cc_library(
    name = "Vectorize",
    srcs = glob([
        "lib/Transforms/Vectorize/*.cpp",
        "lib/Transforms/Vectorize/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Transforms/Vectorize/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Analysis",
        ":Core",
        ":Support",
        ":TransformUtils",
    ],
)

cc_library(
    name = "ipo",
    srcs = glob([
        "lib/Transforms/IPO/*.cpp",
        "lib/Transforms/IPO/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Transforms/IPO/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":AggressiveInstCombine",
        ":Analysis",
        ":BitReader",
        ":BitWriter",
        ":Core",
        ":IRReader",
        ":InstCombine",
        ":Instrumentation",
        ":Linker",
        ":Object",
        ":ProfileData",
        ":Scalar",
        ":Support",
        ":TransformUtils",
        ":Vectorize",
    ],
)

cc_library(
    name = "Analysis",
    srcs = glob([
        "lib/Analysis/*.cpp",
        "lib/Analysis/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Analysis/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":BinaryFormat",
        ":Core",
        ":Object",
        ":ProfileData",
        ":Support",
    ],
)

cc_library(
    name = "ProfileData",
    srcs = glob([
        "lib/ProfileData/*.cpp",
        "lib/ProfileData/*.h",
    ]),
    hdrs = glob([
        "include/llvm/ProfileData/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Core",
        ":Support",
    ],
)

cc_library(
    name = "Target",
    srcs = glob([
        "lib/Target/*.cpp",
        "lib/Target/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Target/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Analysis",
        ":Core",
        ":MC",
        ":Support",
    ],
)

X86_COPTS = select({
    "@toolchain//:windows_x86_64": [
        "/Iexternal/llvm/lib/Target/X86",
        "/I$(GENDIR)/external/llvm/lib/Target/X86",
    ],
    "//conditions:default": [
        "-iquote",
        "external/llvm/lib/Target/X86",
        "-iquote",
        "$(GENDIR)/external/llvm/lib/Target/X86",
    ],
})

cc_library(
    name = "X86AsmParser",
    srcs = glob([
        "lib/Target/X86/AsmParser/*.cpp",
        "lib/Target/X86/AsmParser/*.h",
    ]) + [
        ":gen-asm-matcher",
    ],
    hdrs = glob([
        "include/llvm/Target/X86/AsmParser/*.h",
    ]),
    copts = PLATFORM_COPTS + X86_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":MC",
        ":MCParser",
        ":Support",
        ":X86Desc",
        ":X86Info",
    ],
)

cc_library(
    name = "X86",
    visibility = ["//visibility:public"],
    deps = [
        ":X86AsmParser",
        ":X86CodeGen",
    ],
)

cc_library(
    name = "X86CodeGen",
    srcs = glob([
        "lib/Target/X86/*.cpp",
        "lib/Target/X86/*.h",
    ]) + [
        ":gen-callingconv",
        ":gen-fast-isel",
        ":gen-x86-evex2vex-tables",
        ":gen-register-bank",
        ":gen-dag-isel",
        ":gen-global-isel",
    ],
    hdrs = glob([
        "include/llvm/Target/X86/*.h",
        "lib/Target/X86/*.def",
    ]),
    copts = PLATFORM_COPTS + X86_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Analysis",
        ":AsmPrinter",
        ":CodeGen",
        ":Core",
        ":GlobalISel",
        ":MC",
        ":ProfileData",
        ":SelectionDAG",
        ":Support",
        ":Target",
        ":X86Desc",
        ":X86Info",
        ":X86Utils",
    ],
)

cc_library(
    name = "X86Desc",
    srcs = glob([
        "lib/Target/X86/MCTargetDesc/*.cpp",
        "lib/Target/X86/MCTargetDesc/*.h",
    ]) + [
        ":gen-register-info",
        ":gen-instr-info",
        ":gen-subtarget",
        ":gen-asm-writer",
        ":gen-asm-writer-1",
    ],
    hdrs = glob([
        "include/llvm/Target/X86/MCTargetDesc/*.h",
    ]),
    copts = PLATFORM_COPTS + X86_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":MC",
        ":MCDisassembler",
        ":Object",
        ":Support",
        ":X86Info",
        ":X86Utils",
    ],
)

cc_library(
    name = "X86Info",
    srcs = glob([
        "lib/Target/X86/TargetInfo/*.cpp",
        "lib/Target/X86/TargetInfo/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Target/X86/TargetInfo/*.h",
    ]),
    copts = PLATFORM_COPTS + X86_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Support",
    ],
)

cc_library(
    name = "X86Utils",
    srcs = glob([
        "lib/Target/X86/Utils/*.cpp",
        "lib/Target/X86/Utils/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Target/X86/Utils/*.h",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":Support",
    ],
)

cc_binary(
    name = "llvm-tblgen",
    srcs = glob([
        "utils/TableGen/**/*.cpp",
        "utils/TableGen/**/*.h",
    ]),
    copts = PLATFORM_COPTS,
    linkopts = select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [":TableGen"],
)

tblgen(
    name = "gen-attrs",
    src = "include/llvm/IR/Attributes.td",
    out = "include/llvm/IR/Attributes.inc",
    action = "-gen-attrs",
    incs = ["include"],
)

tblgen(
    name = "gen-attrs-compat",
    src = "lib/IR/AttributesCompatFunc.td",
    out = "lib/IR/AttributesCompatFunc.inc",
    action = "-gen-attrs",
    incs = ["include"],
)

tblgen(
    name = "gen-intrinsic-enums",
    src = "include/llvm/IR/Intrinsics.td",
    out = "include/llvm/IR/IntrinsicEnums.inc",
    action = "-gen-intrinsic-enums",
    incs = ["include"],
)

tblgen(
    name = "gen-intrinsic-impl",
    src = "include/llvm/IR/Intrinsics.td",
    out = "include/llvm/IR/IntrinsicImpl.inc",
    action = "-gen-intrinsic-impl",
    incs = ["include"],
)

tblgen(
    name = "gen-disassembler",
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenDisassemblerTables.inc",
    action = "-gen-disassembler",
    incs = [
        "include",
        "lib/Target/X86",
    ],
)

tblgen(
    name = "gen-instr-info",
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenInstrInfo.inc",
    action = "-gen-instr-info",
    incs = [
        "include",
        "lib/Target/X86",
    ],
)

tblgen(
    name = "gen-register-info",
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenRegisterInfo.inc",
    action = "-gen-register-info",
    incs = [
        "include",
        "lib/Target/X86",
    ],
)

tblgen(
    name = "gen-register-bank",
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenRegisterBank.inc",
    action = "-gen-register-bank",
    incs = [
        "include",
        "lib/Target/X86",
    ],
)

tblgen(
    name = "gen-subtarget",
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenSubtargetInfo.inc",
    action = "-gen-subtarget",
    incs = [
        "include",
        "lib/Target/X86",
    ],
)

tblgen(
    name = "gen-asm-writer",
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenAsmWriter.inc",
    action = "-gen-asm-writer",
    incs = [
        "include",
        "lib/Target/X86",
    ],
)

tblgen(
    name = "gen-asm-writer-1",
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenAsmWriter1.inc",
    action = "-gen-asm-writer",
    flags = ["-asmwriternum=1"],
    incs = [
        "include",
        "lib/Target/X86",
    ],
)

tblgen(
    name = "gen-asm-matcher",
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenAsmMatcher.inc",
    action = "-gen-asm-matcher",
    incs = [
        "include",
        "lib/Target/X86",
    ],
)

tblgen(
    name = "gen-dag-isel",
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenDAGISel.inc",
    action = "-gen-dag-isel",
    incs = [
        "include",
        "lib/Target/X86",
    ],
)

tblgen(
    name = "gen-callingconv",
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenCallingConv.inc",
    action = "-gen-callingconv",
    incs = [
        "include",
        "lib/Target/X86",
    ],
)

tblgen(
    name = "gen-fast-isel",
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenFastISel.inc",
    action = "-gen-fast-isel",
    incs = [
        "include",
        "lib/Target/X86",
    ],
)

tblgen(
    name = "gen-global-isel",
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenGlobalISel.inc",
    action = "-gen-global-isel",
    incs = [
        "include",
        "lib/Target/X86",
    ],
)

tblgen(
    name = "gen-x86-evex2vex-tables",
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenEVEX2VEXTables.inc",
    action = "-gen-x86-EVEX2VEX-tables",
    incs = [
        "include",
        "lib/Target/X86",
    ],
)

tblgen(
    name = "gen-inst-combine-tables",
    src = "lib/Transforms/InstCombine/InstCombineTables.td",
    out = "lib/Transforms/InstCombine/InstCombineTables.inc",
    action = "-gen-searchable-tables",
    incs = ["include"],
)

# A creator of an empty file include/llvm/Support/VCSRevision.h.
# This would be constructed by the CMake build infrastructure; a few of the
# source files try to include it. We leave it blank.
genrule(
    name = "llvm_vcsrevision_h",
    srcs = [],
    outs = ["include/llvm/Support/VCSRevision.h"],
    cmd = "echo '#define LLVM_REVISION \"pml-llvm\"' > $@",
)

py_binary(
    name = "lit",
    srcs = glob(["utils/lit/**/*.py"]),
    imports = ["utils/lit"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "FileCheck",
    srcs = glob(["utils/FileCheck/**/*.cpp"]),
    copts = PLATFORM_COPTS,
    linkopts = select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": ["-lm"],
    }),
    visibility = ["//visibility:public"],
    deps = [":Support"],
)

cc_binary(
    name = "count",
    srcs = glob(["utils/count/**/*.c"]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "not",
    srcs = glob(["utils/not/**/*.cpp"]),
    copts = PLATFORM_COPTS,
    linkopts = select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": ["-lm"],
    }),
    visibility = ["//visibility:public"],
    deps = [":Support"],
)

genrule(
    name = "license",
    srcs = ["LICENSE.TXT"],
    outs = ["llvm-LICENSE"],
    cmd = "cp $(SRCS) $@",
    visibility = ["//visibility:public"],
)
