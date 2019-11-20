TBLGEN_ACTIONS = [
    "-gen-op-lib-cpp-wrappers",
    "-gen-op-lib-py-wrappers",
]

COPTS = select({
    "@com_intel_plaidml//toolchain:windows_x86_64": [
        "/wd4624",
    ],
    "//conditions:default": [],
})

def _tblgen_impl(ctx):
    args = ctx.actions.args()
    args.add(ctx.attr.action)
    args.add_all(ctx.attr.flags)
    args.add("-I", ctx.label.workspace_root)
    args.add_all(ctx.files.incs, before_each = "-I")
    args.add("-o", ctx.outputs.out)
    args.add(ctx.file.src)
    ctx.actions.run(
        inputs = [ctx.file.src] + ctx.files.also,
        outputs = [ctx.outputs.out],
        arguments = [args],
        executable = ctx.executable._tool,
        mnemonic = "OpLibTableGen",
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

op_lib_tblgen_rule = rule(
    attrs = {
        "src": attr.label(
            allow_single_file = [".td"],
            mandatory = True,
        ),
        "also": attr.label_list(
            allow_files = [".td"],
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
            default = Label("//pmlc/dialect/op_lib:op-lib-tblgen"),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
    },
    output_to_genfiles = True,
    implementation = _tblgen_impl,
)

def op_lib_tblgen(name, src, out, incs, action, also = [], flags = []):
    op_lib_tblgen_rule(
        name = "%s_rule" % name,
        src = src,
        also = also,
        out = out,
        incs = incs,
        action = action,
        flags = flags,
    )
    native.cc_library(
        name = name,
        textual_hdrs = [out],
    )
