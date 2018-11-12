OUT_DIR = "include/llvm/"

GENFILES = [
    OUT_DIR + "Config/AsmParsers.def",
    OUT_DIR + "Config/AsmPrinters.def",
    OUT_DIR + "Config/config.h",
    OUT_DIR + "Config/Disassemblers.def",
    OUT_DIR + "Config/llvm-config.h",
    OUT_DIR + "Config/Targets.def",
    OUT_DIR + "Support/DataTypes.h",
]

# Local GenRule to configure llvm
genrule(
    name = "configure",
    srcs = ["CMakeLists.txt"],
    outs = GENFILES,
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
    -DHAVE_LIBEDIT=0
""",
        "//conditions:default": """
cmake -B$(@D) -H$$(dirname $(location //:CMakeLists.txt)) \
    -DPYTHON_EXECUTABLE=$$(which python3) \
    -DLLVM_ENABLE_TERMINFO=OFF
""",
    }),
)

TARGETS = {
    "AArch64": "AArch64",
    "AMDGPU": "AMDGPU",
    "ARM": "ARM",
    "AVR": "AVR",
    "BPF": "BPF",
    "CppBackend": "CPPBackend",
    "Hexagon": "Hexagon",
    "Mips": "Mips",
    "MSP430": "MSP430",
    "NVPTX": "NVPTX",
    "PowerPC": "PPC",
    "Sparc": "Sparc",
    "SystemZ": "SystemZ",
    "WebAssembly": "WebAssembly",
    "X86": "X86",
    "XCore": "XCore",
}

TARGET_INCLUDES = " ".join(["-I external/llvm_archive/lib/Target/%s" % (s) for s in TARGETS.keys()])

TBLGEN_CMD_MULTI = """
    for s in $(SRCS); do 
      $(location :tblgen) %s -I external/llvm_archive/include $$s > $(@D)/../../$$(dirname $$s)/$$(basename -s .td $$s)%s; 
    done
"""

TBLGEN_CMD_SINGLE = """
    for s in $(SRCS); do 
      $(location :tblgen) %s -I external/llvm_archive/include $$s > $(@D)/$$(basename -s .td $$s)%s; 
    done
"""

GEN_ATTR_IN = [
    "include/llvm/IR/Attributes.td",
    "lib/IR/AttributesCompatFunc.td",
]

GEN_ATTR_OUT = [s[:-len(".td")] + ".inc" for s in GEN_ATTR_IN]

genrule(
    name = "gen_inc_attrs",
    srcs = GEN_ATTR_IN,
    outs = GEN_ATTR_OUT,
    cmd = TBLGEN_CMD_MULTI % ("-gen-attrs", ".inc"),
    local = True,
    tools = [":tblgen"],
)

GEN_INTRIN_IN = [
    "include/llvm/IR/Intrinsics.td",
]

GEN_INTRIN_OUT = [s[:-len(".td")] + ".gen" for s in GEN_INTRIN_IN]

genrule(
    name = "gen_intrin",
    srcs = GEN_INTRIN_IN,
    outs = GEN_INTRIN_OUT,
    cmd = TBLGEN_CMD_SINGLE % ("-gen-intrinsic", ".gen"),
    local = True,
    tools = [":tblgen"],
)

GEN_OPT_IN = [
    "lib/LibDriver/Options.td",
]

GEN_OPT_OUT = [s[:-len(".td")] + ".inc" for s in GEN_OPT_IN]

genrule(
    name = "gen_opt",
    srcs = GEN_OPT_IN,
    outs = GEN_OPT_OUT,
    cmd = TBLGEN_CMD_SINGLE % ("-gen-opt-parser-defs", ".inc"),
    local = True,
    tools = [":tblgen"],
)

GENS = {
    "reg-info": (
        "GenRegisterInfo.inc",
        ["lib/Target/%s/%s.td" % (k, v) for k, v in TARGETS.items() if k not in ["CppBackend"]],
        "-gen-register-info",
    ),
    "instr-info": (
        "GenInstrInfo.inc",
        ["lib/Target/%s/%s.td" % (k, v) for k, v in TARGETS.items() if k not in [
            "AVR",
            "CppBackend",
        ]],
        "-gen-instr-info",
    ),
    "subtarget-info": (
        "GenSubtargetInfo.inc",
        ["lib/Target/%s/%s.td" % (k, v) for k, v in TARGETS.items() if k not in ["CppBackend"]],
        "-gen-subtarget",
    ),
    "calling-conv": (
        "GenCallingConv.inc",
        ["lib/Target/%s/%s.td" % (k, v) for k, v in TARGETS.items() if k not in [
            "AVR",
            "CppBackend",
        ]],
        "-gen-callingconv",
    ),
    "dag-isl": (
        "GenDAGISel.inc",
        ["lib/Target/%s/%s.td" % (k, v) for k, v in TARGETS.items() if k not in [
            "AVR",
            "CppBackend",
        ]],
        "-gen-dag-isel",
    ),
    "fast-isl": (
        "GenFastISel.inc",
        ["lib/Target/%s/%s.td" % (k, v) for k, v in TARGETS.items() if k not in [
            "AVR",
            "CppBackend",
            "Hexagon",
            "SystemZ",
        ]],
        "-gen-fast-isel",
    ),
    "pseudeo-lowering": (
        "GenMCPseudoLowering.inc",
        ["lib/Target/%s/%s.td" % (k, v) for k, v in TARGETS.items() if k not in ["CppBackend"]],
        "-gen-pseudo-lowering",
    ),
    "asm-matcher": (
        "GenAsmMatcher.inc",
        ["lib/Target/%s/%s.td" % (k, v) for k, v in TARGETS.items() if k not in [
            "CppBackend",
            "MSP430",
            "NVPTX",
            "XCore",
        ]],
        "-gen-asm-matcher",
    ),
    "asm-writer": (
        "GenAsmWriter.inc",
        ["lib/Target/%s/%s.td" % (k, v) for k, v in TARGETS.items() if k not in ["CppBackend"]],
        "-gen-asm-writer",
    ),
    "asm-writer-one": (
        "GenAsmWriter1.inc",
        ["lib/Target/%s/%s.td" % (k, v) for k, v in TARGETS.items() if k in [
            "AArch64",
            "X86",
        ]],
        "-gen-asm-writer -asmwriternum=1",
    ),
    "disassembler-tables": (
        "GenDisassemblerTables.inc",
        ["lib/Target/%s/%s.td" % (k, v) for k, v in TARGETS.items() if k not in [
            "AMDGPU",
            "AVR",
            "BPF",
            "CppBackend",
            "NVPTX",
            "MSP430",
        ]],
        "-gen-disassembler",
    ),
    "code-emitter": (
        "GenMCCodeEmitter.inc",
        ["lib/Target/%s/%s.td" % (k, v) for k, v in TARGETS.items() if k not in [
            "AVR",
            "CppBackend",
            "X86",
        ]],
        "-gen-emitter",
    ),
    "intrinsics": (
        "GenIntrinsics.inc",
        ["lib/Target/%s/%s.td" % (k, v) for k, v in TARGETS.items() if k not in ["CppBackend"]],
        "-gen-intrinsic",
    ),
    "dfa-packetizer": (
        "GenDFAPacketizer.inc",
        ["lib/Target/%s/%s.td" % (k, v) for k, v in TARGETS.items() if k in [
            "AMDGPU",
            "Hexagon",
        ]],
        "-gen-dfa-packetizer",
    ),
}

# Bazel doesn't support loops in BUILD files, and it's not currently possible to load a .bzl file from an external repository, so
# instead we do this. Ugly, should be fixable someday

GEN_CMDS = [
    {
        "name": name,
        "srcs": dat[1],
        "tools": [":tblgen"],
        "outs": [s[:-len(".td")] + dat[0] for s in dat[1]],
        "cmd": TBLGEN_CMD_MULTI % (
            TARGET_INCLUDES + " " + dat[2],
            dat[0],
        ),
    }
    for name, dat in GENS.items()
]

INCS = []

C = GEN_CMDS[0]

genrule(
    name = C["name"],
    srcs = C["srcs"],
    outs = C["outs"],
    cmd = C["cmd"],
    local = True,
    tools = C["tools"],
)

INCS += C["outs"]

C = GEN_CMDS[1]

genrule(
    name = C["name"],
    srcs = C["srcs"],
    outs = C["outs"],
    cmd = C["cmd"],
    local = True,
    tools = C["tools"],
)

INCS += C["outs"]

C = GEN_CMDS[2]

genrule(
    name = C["name"],
    srcs = C["srcs"],
    outs = C["outs"],
    cmd = C["cmd"],
    local = True,
    tools = C["tools"],
)

INCS += C["outs"]

C = GEN_CMDS[3]

genrule(
    name = C["name"],
    srcs = C["srcs"],
    outs = C["outs"],
    cmd = C["cmd"],
    local = True,
    tools = C["tools"],
)

INCS += C["outs"]

C = GEN_CMDS[4]

genrule(
    name = C["name"],
    srcs = C["srcs"],
    outs = C["outs"],
    cmd = C["cmd"],
    local = True,
    tools = C["tools"],
)

INCS += C["outs"]

C = GEN_CMDS[5]

genrule(
    name = C["name"],
    srcs = C["srcs"],
    outs = C["outs"],
    cmd = C["cmd"],
    local = True,
    tools = C["tools"],
)

INCS += C["outs"]

C = GEN_CMDS[6]

genrule(
    name = C["name"],
    srcs = C["srcs"],
    outs = C["outs"],
    cmd = C["cmd"],
    local = True,
    tools = C["tools"],
)

INCS += C["outs"]

C = GEN_CMDS[7]

genrule(
    name = C["name"],
    srcs = C["srcs"],
    outs = C["outs"],
    cmd = C["cmd"],
    local = True,
    tools = C["tools"],
)

INCS += C["outs"]

C = GEN_CMDS[8]

genrule(
    name = C["name"],
    srcs = C["srcs"],
    outs = C["outs"],
    cmd = C["cmd"],
    local = True,
    tools = C["tools"],
)

INCS += C["outs"]

C = GEN_CMDS[9]

genrule(
    name = C["name"],
    srcs = C["srcs"],
    outs = C["outs"],
    cmd = C["cmd"],
    local = True,
    tools = C["tools"],
)

INCS += C["outs"]

C = GEN_CMDS[10]

genrule(
    name = C["name"],
    srcs = C["srcs"],
    outs = C["outs"],
    cmd = C["cmd"],
    local = True,
    tools = C["tools"],
)

INCS += C["outs"]

C = GEN_CMDS[11]

genrule(
    name = C["name"],
    srcs = C["srcs"],
    outs = C["outs"],
    cmd = C["cmd"],
    local = True,
    tools = C["tools"],
)

INCS += C["outs"]

C = GEN_CMDS[12]

genrule(
    name = C["name"],
    srcs = C["srcs"],
    outs = C["outs"],
    cmd = C["cmd"],
    local = True,
    tools = C["tools"],
)

INCS += C["outs"]

C = GEN_CMDS[13]

genrule(
    name = C["name"],
    srcs = C["srcs"],
    outs = C["outs"],
    cmd = C["cmd"],
    local = True,
    tools = C["tools"],
)

INCS += C["outs"]

# Valgrind is causing cross-compilation issues and there's no way to disable it so we stub it out
genrule(
    name = "workaround_valgrind",
    outs = ["ValgrindStub.cpp"],
    cmd = """
cat <<'EOF' >$@
#include "llvm/Support/Valgrind.h"
#include "llvm/Config/config.h"
#include <cstddef>

bool llvm::sys::RunningOnValgrind() {
  return false;
}

void llvm::sys::ValgrindDiscardTranslations(const void *Addr, size_t Len) {
}

EOF
""",
)

PLATFORM_COPTS = select({
    "@toolchain//:macos_x86_64": [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
        "-w",
    ],
    "//conditions:default": [
        "-w",
        "-std=c++1y",
    ],
})

cc_library(
    name = "base",
    srcs = glob(
        [
            "lib/Support/**/*.cpp",
            "lib/Support/**/*.c",
            "lib/Support/**/*.h",
            "lib/Target/**/*.h",
        ],
        exclude = ["lib/Support/Valgrind.cpp"],
    ) + [
        "ValgrindStub.cpp",
    ],
    hdrs = glob([
        "include/llvm/**/*.def",
        "lib/Support/**/*.inc",
        "include/llvm-c/**/*.h",
        "include/llvm/**/*.h",
    ]) + GENFILES,
    copts = [
        "-fPIC",
    ] + PLATFORM_COPTS,
    includes = [
        "include",
        ":include",
    ],
    linkopts = select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": [
            "-lpthread",
            "-ldl",
        ],
    }),
    deps = [
        "@zlib",
    ],
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
    linkopts = [
        "-lm",
        "-lpthread",
        "-ldl",
    ],
    deps = [":base"],
)

TARGET_INCLUDES = [[
    "-iquote",
    "$(GENDIR)/external/llvm_archive/lib/Target/%s" % (s),
    "-iquote",
    "external/llvm_archive/lib/Target/%s" % (s),
] for s in TARGETS.keys()]

TARGET_INCLUDES = [item for sublist in TARGET_INCLUDES for item in sublist]

cc_library(
    name = "lib",
    srcs = glob(
        [
            "lib/**/*.cpp",
            "lib/**/*.h",
        ],
        exclude = [
            "lib/Support/*",
            "lib/TableGen/*",
            "lib/Target/**/*",
        ] + [
            "lib/DebugInfo/PDB/DIA/**/*",  # need to switch on windows in order to get PDBs
        ] + [
            "lib/ExecutionEngine/OProfileJIT/**/*",
            "lib/ExecutionEngine/IntelJITEvents/**/*",
            "lib/Fuzzer/**/*",  # excluded because they don't build cleanly
        ],
    ) + GEN_ATTR_OUT + GEN_OPT_OUT + INCS,
    hdrs = GEN_INTRIN_OUT + glob([
        "lib/**/*.def",
        "lib/**/*.inc",
        "include/**/*.inc",
    ]),
    copts = [
        "-fPIC",
        "-iquote",
        "$(GENDIR)/external/llvm_archive/lib/LibDriver",
        "-iquote",
        "$(GENDIR)/external/llvm_archive/lib/IR",
    ] + PLATFORM_COPTS,
    includes = ["include"],
    linkopts = select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": ["-lpthread"],
    }),
    deps = [":base"],
    alwayslink = 1,
)

cc_library(
    name = "targets",
    srcs = glob(
        [
            "lib/Target/**/*.cpp",
            "lib/Target/**/*.h",
        ],
        exclude = [
            "lib/Target/XCore/**/*",
            "lib/Target/AMDGPU/**/*",  # excluded because they don't build cleanly
        ] + [
            "lib/Target/AVR/**/*",
            "lib/Target/BPF/**/*",
            "lib/Target/Mips/**/*",
            "lib/Target/MP430/**/*",
            "lib/Target/PowerPC/**/*",
            "lib/Target/Sparc/**/*",
            "lib/Target/SystemZ/**/*",  # excluded for build speed
        ],
    ) + GEN_ATTR_OUT + INCS,
    hdrs = GEN_INTRIN_OUT + glob(["lib/**/*.def"]),
    copts = TARGET_INCLUDES + [
        "-fPIC",
        "-iquote",
        "$(GENDIR)/external/llvm_archive/lib/IR",
    ] + PLATFORM_COPTS,
    includes = ["include"],
    deps = [":base"],
    alwayslink = 1,
)

cc_library(
    name = "llvm",
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":lib",
        ":targets",
    ],
)

# This is a dummy target used for eliciting the static libraries created by ":llvm".
# See @com_intel_plaidml//lib/README.md
cc_binary(
    name = "static.so",
    linkshared = 1,
    deps = [":llvm"],
)

cc_library(
    name = "llvm_inc",
    hdrs = glob([
        "include/llvm/**/*.def",
        "lib/Support/**/*.inc",
        "include/llvm-c/**/*.h",
        "include/llvm/**/*.h",
    ]) + GENFILES + GEN_ATTR_OUT + GEN_INTRIN_OUT,
    includes = [
        "include",
    ],
    visibility = ["//visibility:public"],
)
