def _get_main(ctx):
    if ctx.file.main:
        return ctx.workspace_name + "/" + ctx.file.main.path
    main = ctx.label.name + ".py"
    for src in ctx.files.srcs:
        if src.basename == main:
            return ctx.workspace_name + "/" + src.path
    fail(
        "corresponding default '{}' does not appear in srcs. ".format(main) +
        "Add it or override default file name with a 'main' attribute",
    )

def _conda_impl(ctx):
    env = ctx.attr.env
    script = ctx.actions.declare_file(ctx.label.name)
    main = _get_main(ctx)
    ctx.actions.expand_template(
        template = ctx.file._template,
        output = script,
        substitutions = {
            "%imports%": "",
            "%import_all%": "True",
            "%main%": main,
            "%workspace_name%": ctx.workspace_name,
        },
        is_executable = True,
    )
    runfiles = ctx.runfiles(
        collect_data = True,
        collect_default = True,
        files = ctx.files.srcs,
        root_symlinks = {".cenv": env.files.to_list()[0]},
    )
    return [DefaultInfo(executable = script, runfiles = runfiles)]

_conda_attrs = {
    "srcs": attr.label_list(allow_files = [".py"]),
    "data": attr.label_list(allow_files = True),
    "deps": attr.label_list(),
    "env": attr.label(
        mandatory = True,
        allow_files = True,
    ),
    "main": attr.label(allow_single_file = [".py"]),
    "_template": attr.label(
        default = Label("//bzl:conda.tpl.py"),
        allow_single_file = True,
    ),
}

conda_binary = rule(
    attrs = _conda_attrs,
    executable = True,
    implementation = _conda_impl,
)

conda_test = rule(
    attrs = _conda_attrs,
    test = True,
    implementation = _conda_impl,
)
