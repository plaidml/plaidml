TBLGEN_ACTIONS = [
    "-gen-enum-defs",
    "-gen-enum-decls",
    "-gen-llvmir-conversions",
    "-gen-op-decls",
    "-gen-op-defs",
    "-gen-op-doc",
    "-gen-reference-implementations",
    "-gen-rewriters",
]

def _tblgen_impl(ctx):
    args = ctx.actions.args()
    args.add(ctx.attr.action)
    args.add_all(ctx.attr.flags)
    args.add("-I", ctx.label.workspace_root)
    args.add_all(ctx.files.incs, before_each = "-I")
    args.add("-o", ctx.outputs.out)
    args.add(ctx.file.src)
    ctx.actions.run(
        inputs = [ctx.file.src],
        outputs = [ctx.outputs.out],
        arguments = [args],
        executable = ctx.executable._tool,
        mnemonic = "MLIRTableGen",
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

mlir_tblgen = rule(
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
            default = Label("@mlir//:mlir-tblgen"),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
    },
    output_to_genfiles = True,
    implementation = _tblgen_impl,
)
