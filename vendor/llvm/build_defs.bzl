TBLGEN_ACTIONS = [
    "-gen-asm-matcher",
    "-gen-asm-writer",
    "-gen-attrs",
    "-gen-callingconv",
    "-gen-compress-inst-emitter",
    "-gen-ctags",
    "-gen-dag-isel",
    "-gen-dfa-packetizer",
    "-gen-disassembler",
    "-gen-emitter",
    "-gen-fast-isel",
    "-gen-global-isel",
    "-gen-intrinsic-enums",
    "-gen-intrinsic-impl",
    "-gen-instr-docs",
    "-gen-instr-info",
    "-gen-opt-parser-defs",
    "-gen-pseudo-lowering",
    "-gen-register-bank",
    "-gen-register-info",
    "-gen-searchable-tables",
    "-gen-subtarget",
    "-gen-tgt-intrinsic-enums",
    "-gen-tgt-intrinsic-impl",
    "-gen-x86-EVEX2VEX-tables",
    "-gen-x86-fold-tables",
    "-help",
]

def _tblgen_impl(ctx):
    args = ctx.actions.args()
    args.add(ctx.attr.action)
    args.add_all(ctx.attr.flags)
    args.add_all(ctx.files.incs, before_each = "-I")
    args.add("-o", ctx.outputs.out)
    args.add(ctx.file.src)
    ctx.actions.run(
        inputs = [ctx.file.src],
        outputs = [ctx.outputs.out],
        arguments = [args],
        executable = ctx.executable._tool,
        mnemonic = "TableGen",
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

tblgen = rule(
    attrs = {
        "src": attr.label(
            allow_single_file = [".td"],
            mandatory = True,
        ),
        "out": attr.output(
            mandatory = True,
        ),
        "incs": attr.label_list(
            allow_files = True,
        ),
        "action": attr.string(
            mandatory = True,
            values = TBLGEN_ACTIONS,
        ),
        "flags": attr.string_list(),
        "_tool": attr.label(
            default = Label("@llvm//:llvm-tblgen"),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
    },
    output_to_genfiles = True,
    implementation = _tblgen_impl,
)
