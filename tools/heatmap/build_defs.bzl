def _heatmap_impl(ctx):
    args = ctx.actions.args()
    args.add(ctx.file.csv)
    args.add(ctx.file.template)
    args.add(ctx.outputs.out)

    ctx.actions.run(
        mnemonic = "heatmap",
        executable = ctx.executable._tool,
        arguments = [args],
        inputs = [ctx.file.csv, ctx.file.template],
        outputs = [ctx.outputs.out],
        tools = [ctx.executable._tool],
    )

    return DefaultInfo(files = depset([ctx.outputs.out]))

heatmap = rule(
    attrs = {
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
    },
    implementation = _heatmap_impl,
)
