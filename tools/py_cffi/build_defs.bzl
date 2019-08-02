def _py_cffi_impl(ctx):
    args = ctx.actions.args()
    args.add_all(ctx.files.srcs, before_each = "--source")
    args.add("--module", ctx.attr.module)
    args.add("--output", ctx.outputs.out)
    ctx.actions.run(
        inputs = [ctx.executable._tool] + ctx.files.srcs,
        outputs = [ctx.outputs.out],
        arguments = [args],
        executable = ctx.executable._tool,
        mnemonic = "PyCffi",
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

py_cffi_rule = rule(
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "module": attr.string(
            mandatory = True,
        ),
        "out": attr.output(),
        "_tool": attr.label(
            default = Label("//tools/py_cffi"),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
    },
    implementation = _py_cffi_impl,
)

# It's named srcs_ordered because we want to prevent buildifier from automatically sorting this list.
def py_cffi(name, module, srcs_ordered, tags = [], **kwargs):
    out = name + ".py"
    py_cffi_rule(
        name = name + "_py_cffi",
        module = module,
        srcs = srcs_ordered,
        out = out,
        tags = tags + ["conda"],
    )

    native.py_library(
        name = name,
        srcs = [out],
        tags = tags + ["conda"],
        **kwargs
    )
