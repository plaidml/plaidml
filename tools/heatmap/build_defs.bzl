load("//bzl:python.bzl", "run_python_attrs", "run_python_tool")

def _heatmap_impl(ctx):
    args = ctx.actions.args()
    args.add(ctx.file.csv)
    args.add(ctx.file.template)
    args.add(ctx.outputs.out)

    run_python_tool(
        ctx,
        mnemonic = "heatmap",
        python = ctx.file._python,
        tool = ctx.executable._tool,
        args = args,
        inputs = [ctx.file.csv, ctx.file.template],
        outputs = [ctx.outputs.out],
    )

    return DefaultInfo(files = depset([ctx.outputs.out]))

heatmap = rule(
    attrs = run_python_attrs({
        "_tool": attr.label(
            default = Label("//tools/heatmap"),
            executable = True,
            cfg = "host",
        ),
        "csv": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "template": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "out": attr.output(),
    }),
    implementation = _heatmap_impl,
)
