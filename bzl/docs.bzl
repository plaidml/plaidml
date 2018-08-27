def _impl(ctx):
    srcdir = ctx.file.conf.dirname
    out = ctx.outputs.out
    ctx.actions.run(
        inputs = ctx.files.srcs,
        outputs = [out],
        executable = ctx.file.sphinx,
        arguments = [srcdir, out.path, '--plantuml', ctx.file._plantuml.path],
        use_default_shell_env = True,
        mnemonic = "SphinxBuild",
    )
    return DefaultInfo(files=depset([out]))

sphinx = rule(
    attrs = {
        "srcs": attr.label_list(
            mandatory = True,
            allow_files = True,
        ),
        "conf": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "out": attr.output(mandatory = True),
        "sphinx": attr.label(
            default = Label("//bzl:sphinx"),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
        "_plantuml": attr.label(
            default = Label("@plantuml_jar//file"),
            allow_single_file = True,
        ),
    },
    implementation = _impl,
)
