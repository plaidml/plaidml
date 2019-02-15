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
    action = "-gen-attrs",
    incs = ["include"],
    src = "include/llvm/IR/Attributes.td",
    out = "include/llvm/IR/Attributes.inc",
)

tblgen(
    name = "gen-attrs-compat",
    action = "-gen-attrs",
    incs = ["include"],
    src = "lib/IR/AttributesCompatFunc.td",
    out = "lib/IR/AttributesCompatFunc.inc",
)

tblgen(
    name = "gen-intrinsic-enums",
    action = "-gen-intrinsic-enums",
    incs = ["include"],
    src = "include/llvm/IR/Intrinsics.td",
    out = "include/llvm/IR/IntrinsicEnums.inc",
)

tblgen(
    name = "gen-intrinsic-impl",
    action = "-gen-intrinsic-impl",
    incs = ["include"],
    src = "include/llvm/IR/Intrinsics.td",
    out = "include/llvm/IR/IntrinsicImpl.inc",
)

tblgen(
    name = "gen-disassembler",
    action = "-gen-disassembler",
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenDisassemblerTables.inc",
)

tblgen(
    name = "gen-instr-info",
    action = "-gen-instr-info",
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenInstrInfo.inc",
)

tblgen(
    name = "gen-register-info",
    action = "-gen-register-info",
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenRegisterInfo.inc",
)

tblgen(
    name = "gen-register-bank",
    action = "-gen-register-bank",
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenRegisterBank.inc",
)

tblgen(
    name = "gen-subtarget",
    action = "-gen-subtarget",
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenSubtargetInfo.inc",
)

tblgen(
    name = "gen-asm-writer",
    action = "-gen-asm-writer",
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenAsmWriter.inc",
)

tblgen(
    name = "gen-asm-writer-1",
    action = "-gen-asm-writer",
    flags = ["-asmwriternum=1"],
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenAsmWriter1.inc",
)

tblgen(
    name = "gen-asm-matcher",
    action = "-gen-asm-matcher",
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenAsmMatcher.inc",
)

tblgen(
    name = "gen-dag-isel",
    action = "-gen-dag-isel",
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenDAGISel.inc",
)

tblgen(
    name = "gen-callingconv",
    action = "-gen-callingconv",
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenCallingConv.inc",
)

tblgen(
    name = "gen-fast-isel",
    action = "-gen-fast-isel",
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenFastISel.inc",
)

tblgen(
    name = "gen-global-isel",
    action = "-gen-global-isel",
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenGlobalISel.inc",
)

tblgen(
    name = "gen-x86-evex2vex-tables",
    action = "-gen-x86-EVEX2VEX-tables",
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/Target/X86/X86.td",
    out = "lib/Target/X86/X86GenEVEX2VEXTables.inc",
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
    deps = [
        ":support",
    ],
    copts = PLATFORM_COPTS + [
        "-iquote",
        "external/llvm/lib/Target/X86",
        "-iquote",
        "$(GENDIR)/external/llvm/lib/Target/X86",
    ],
    alwayslink = 1,
)

tblgen(
    name = "gen-lib-opt-parser-defs",
    action = "-gen-opt-parser-defs",
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/ToolDrivers/llvm-lib/Options.td",
    out = "lib/ToolDrivers/llvm-lib/Options.inc",
)

tblgen(
    name = "gen-dlltool-opt-parser-defs",
    action = "-gen-opt-parser-defs",
    incs = [
        "include",
        "lib/Target/X86",
    ],
    src = "lib/ToolDrivers/llvm-dlltool/Options.td",
    out = "lib/ToolDrivers/llvm-dlltool/Options.inc",
)

tblgen(
    name = "gen-inst-combine-tables",
    action = "-gen-searchable-tables",
    incs = ["include"],
    src = "lib/Transforms/InstCombine/InstCombineTables.td",
    out = "lib/Transforms/InstCombine/InstCombineTables.inc",
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
        ] + [
            # need to switch on windows in order to get PDBs
            "lib/DebugInfo/PDB/DIA/**/*",
        ] + [
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
        "include/llvm/Support/VCSRevision.h",
    ] + [
        ":gen-attrs",
        ":gen-intrinsic-impl",
        ":gen-intrinsic-enums",
        ":gen-inst-combine-tables",
    ],
    deps = [
        ":support",
    ],
    copts = PLATFORM_COPTS + [
        "-iquote",
        "$(GENDIR)/external/llvm/lib/IR",
        "-iquote",
        "$(GENDIR)/external/llvm/lib/Transforms/InstCombine",
    ],
    alwayslink = 1,
)

cc_library(
    name = "llvm",
    visibility = ["//visibility:public"],
    deps = [
        ":support",
        ":lib",
        ":targets",
    ],
)

# This is used for pre-compiled libraries
cc_library(
    name = "inc",
    visibility = ["//visibility:public"],
    hdrs = glob([
        "lib/Support/**/*.inc",
        "lib/Target/**/*.h",
    ]) + CFG_FILES + [
        ":gen-attrs",
        ":gen-intrinsic-impl",
        ":gen-intrinsic-enums",
    ],
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
)

# This is a dummy target used for eliciting the static libraries created by ":llvm".
# See @com_intel_plaidml//vendor/llvm/lib/README.md
cc_binary(
    name = "static.so",
    linkshared = 1,
    deps = [":llvm"],
)

genrule(
    name = "license",
    visibility = ["//visibility:public"],
    srcs = ["LICENSE.TXT"],
    outs = ["llvm-LICENSE"],
    cmd = "cp $(SRCS) $@",
)
