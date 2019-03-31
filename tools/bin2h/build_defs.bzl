# Copyright 2019 Intel Corporation.

def _bin2h_impl(ctx):
    args = ctx.actions.args()
    args.add(ctx.executable._tool)
    for target, symbol in ctx.attr.srcs.items():
        for src in target.files:
            args.add("--input", "{}={}".format(symbol, src.path))
    args.add("--output", ctx.outputs.out)
    ctx.actions.run(
        inputs = [ctx.executable._tool] + ctx.files.srcs,
        outputs = [ctx.outputs.out],
        arguments = [args],
        executable = "python",
        use_default_shell_env = True,
        mnemonic = "bin2h",
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

bin2h = rule(
    attrs = {
        "srcs": attr.label_keyed_string_dict(
            allow_files = True,
            mandatory = True,
        ),
        "_tool": attr.label(
            default = Label("//tools/bin2h"),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
    },
    implementation = _bin2h_impl,
    output_to_genfiles = True,
    outputs = {"out": "%{name}.h"},
)
