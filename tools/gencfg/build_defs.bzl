# Copyright 2019 Intel Corporation.

_formats = {
    "json": "%{name}.config.json",
    "protobuf": "%{name}.config.pb",
    "prototxt": "%{name}.config.prototxt",
}

def _gencfg_output(format):
    return {"out": _formats[format]}

def _gencfg_impl(ctx):
    args = ctx.actions.args()
    args.add(ctx.executable._tool)
    args.add(ctx.file.tmpl)
    args.add("-y", ctx.file.yml)
    args.add("-t", ctx.attr.target)
    args.add("-f", ctx.attr.format)
    args.add("-o", ctx.outputs.out)
    ctx.actions.run(
        inputs = [ctx.executable._tool, ctx.file.tmpl, ctx.file.yml],
        outputs = [ctx.outputs.out],
        arguments = [args],
        executable = "python",
        use_default_shell_env = True,
        mnemonic = "gencfg",
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

gencfg = rule(
    attrs = {
        "tmpl": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "yml": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "target": attr.string(
            mandatory = True,
        ),
        "format": attr.string(
            default = "protobuf",
            values = _formats.keys(),
        ),
        "_tool": attr.label(
            default = Label("//tools/gencfg"),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
    },
    implementation = _gencfg_impl,
    outputs = _gencfg_output,
)
