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
        "@toolchain//:linux_arm_32v7": """
cmake -B$(@D) -H$$(dirname $(location //:CMakeLists.txt)) \
    -DPYTHON_EXECUTABLE=$$(which python3) \
    -DCMAKE_CROSSCOMPILING=True \
    -DLLVM_DEFAULT_TARGET_TRIPLE=arm-linux-gnueabihf \
    -DLLVM_ENABLE_TERMINFO=OFF
""",
        "@toolchain//:linux_arm_64v8": """
cmake -B$(@D) -H$$(dirname $(location //:CMakeLists.txt)) \
    -DPYTHON_EXECUTABLE=$$(which python3) \
    -DCMAKE_CROSSCOMPILING=True \
    -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-linux-gnueabi \
    -DLLVM_ENABLE_TERMINFO=OFF
""",
        "@toolchain//:macos_x86_64": """
cmake -B$(@D) -H$$(dirname $(location //:CMakeLists.txt)) \
    -DPYTHON_EXECUTABLE=$$(which python3) \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DHAVE_LIBEDIT=0 \
    -DHAVE_FUTIMENS=0 \
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
    name = "support",
    srcs = glob([
        "lib/Support/**/*.cpp",
        "lib/Support/**/*.c",
        "lib/Support/**/*.h",
        "lib/Demangle/**/*.cpp",
        "lib/Demangle/**/*.c",
        "lib/Demangle/**/*.h",
    ]),
    hdrs = glob([
        "lib/Support/**/*.inc",
        "lib/Target/**/*.h",
        "lib/Demangle/**/*.h",
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
    deps = ["@zlib"],
    alwayslink = 1,
)

cc_binary(
    name = "tblgen",
    srcs = glob([
        "utils/TableGen/**/*.cpp",
        "utils/TableGen/**/*.c",
        "utils/TableGen/**/*.h",
        "lib/TableGen/**/*.cpp",
        "lib/TableGen/**/*.c",
        "lib/TableGen/**/*.h",
    ]),
    copts = PLATFORM_COPTS,
    linkopts = ["-lm"],
    deps = [":support"],
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

cc_library(
    name = "targets",
    srcs = glob([
        "lib/Target/*.cpp",
        "lib/Target/*.h",
        "lib/Target/X86/**/*.cpp",
    ]) + [
        ":gen-asm-matcher",
        ":gen-asm-writer",
        ":gen-asm-writer-1",
        ":gen-callingconv",
        ":gen-dag-isel",
        ":gen-disassembler",
        ":gen-fast-isel",
        ":gen-global-isel",
        ":gen-instr-info",
        ":gen-register-info",
        ":gen-register-bank",
        ":gen-subtarget",
        ":gen-x86-evex2vex-tables",
    ],
    hdrs = glob([
        "lib/Target/X86/**/*.def",
    ]) + [
        ":gen-attrs",
        ":gen-intrinsic-impl",
        ":gen-intrinsic-enums",
    ],
    copts = PLATFORM_COPTS + [
        "-iquote",
        "external/llvm/lib/Target/X86",
        "-iquote",
        "$(GENDIR)/external/llvm/lib/Target/X86",
    ],
    deps = [
        ":support",
    ],
    alwayslink = 1,
)

tblgen(
    name = "gen-lib-opt-parser-defs",
    src = "lib/ToolDrivers/llvm-lib/Options.td",
    out = "lib/ToolDrivers/llvm-lib/Options.inc",
    action = "-gen-opt-parser-defs",
    incs = [
        "include",
        "lib/Target/X86",
    ],
)

tblgen(
    name = "gen-dlltool-opt-parser-defs",
    src = "lib/ToolDrivers/llvm-dlltool/Options.td",
    out = "lib/ToolDrivers/llvm-dlltool/Options.inc",
    action = "-gen-opt-parser-defs",
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
    name = "vcs_revision_gen",
    srcs = [],
    outs = ["include/llvm/Support/VCSRevision.h"],
    cmd = "echo '' > \"$@\"",
)

cc_library(
    name = "lib",
    srcs = glob(
        [
            "lib/**/*.cpp",
            "lib/**/*.h",
        ],
        exclude = [
            "lib/Support/**/*",
            "lib/TableGen/*",
            "lib/Target/**/*",
            "lib/ToolDrivers/**/*",
            "lib/Demangle/**/*",
            # need to switch on windows in order to get PDBs
            "lib/DebugInfo/PDB/DIA/**/*",
            # excluded because they don't build cleanly
            "lib/ExecutionEngine/OProfileJIT/**/*",
            "lib/ExecutionEngine/IntelJITEvents/**/*",
            "lib/ExecutionEngine/PerfJITEvents/**/*",
            "lib/Fuzzer/**/*",
            "lib/Testing/**/*",
            "lib/WindowsManifest/**/*",
        ],
    ) + [
        ":gen-attrs-compat",
        ":gen-lib-opt-parser-defs",
        ":gen-dlltool-opt-parser-defs",
    ],
    hdrs = glob([
        "lib/**/*.def",
        "include/llvm/**/*.h",
    ]) + [
        ":vcs_revision_gen",
        ":gen-attrs",
        ":gen-intrinsic-impl",
        ":gen-intrinsic-enums",
        ":gen-inst-combine-tables",
    ],
    copts = PLATFORM_COPTS + [
        "-iquote",
        "$(GENDIR)/external/llvm/lib/IR",
        "-iquote",
        "$(GENDIR)/external/llvm/lib/Transforms/InstCombine",
    ],
    deps = [
        ":support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "llvm",
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
        ":support",
        ":targets",
    ],
)

pkg_tar(
    name = "pkg",
    srcs = glob([
        "include/**",
        "lib/Support/**/*.inc",
        "lib/Target/**/*.h",
    ]) + CFG_FILES + [
        ":gen-attrs",
        ":gen-intrinsic-impl",
        ":gen-intrinsic-enums",
        ":license",
        ":lib",
        ":support",
        ":targets",
    ],
    extension = "tar.bz2",
    strip_prefix = ".",
    visibility = ["//visibility:public"],
)

genrule(
    name = "license",
    srcs = ["LICENSE.TXT"],
    outs = ["llvm-LICENSE"],
    cmd = "cp $(SRCS) $@",
    visibility = ["//visibility:public"],
)
