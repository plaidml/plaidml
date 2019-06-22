# Copyright 2019 Intel Corporation.

def _gencfg_impl(ctx):
    args = ctx.actions.args()
    args.add_all(ctx.files.srcs)
    args.add("--identifier", ctx.attr.identifier)
    args.add("--out", ctx.outputs.out)
    ctx.actions.run(
        inputs = [ctx.executable._tool] + ctx.files.srcs,
        outputs = [ctx.outputs.out],
        arguments = [args],
        executable = ctx.executable._tool,
        mnemonic = "gencfg",
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

gencfg = rule(
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "identifier": attr.string(
            mandatory = True,
        ),
        "_tool": attr.label(
            default = Label("//tools/gencfg"),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
    },
    implementation = _gencfg_impl,
    outputs = {"out": "%{name}.h"},
)
