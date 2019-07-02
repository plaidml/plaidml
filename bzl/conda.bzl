def _get_main(ctx):
    if ctx.file.main:
        return ctx.file.main.path
    main = ctx.label.name + ".py"
    for src in ctx.files.srcs:
        if src.basename == main:
            return src.path
    fail(
        "corresponding default '{}' does not appear in srcs. ".format(main) +
        "Add it or override default file name with a 'main' attribute",
    )

def _conda_impl(ctx):
    env = ctx.attr.env
    launcher = ctx.actions.declare_file(ctx.label.name)
    args = ctx.actions.args()
    args.add(ctx.file.launcher)
    args.add(launcher)
    ctx.actions.run_shell(
        inputs = [ctx.file.launcher],
        outputs = [launcher],
        arguments = [args],
        command = "cp $1 $2",
    )

    launcher_main = ctx.actions.declare_file(ctx.label.name + ".main")
    ctx.actions.write(
        output = launcher_main,
        content = _get_main(ctx),
    )

    runfiles = ctx.runfiles(
        collect_data = True,
        collect_default = True,
        files = ctx.files.srcs,
        symlinks = {
            ".main": launcher_main,
        },
        root_symlinks = {
            ".cenv": env.files.to_list()[0],
            ".main": launcher_main,
        },
    )
    return [DefaultInfo(executable = launcher, runfiles = runfiles)]

_conda_attrs = {
    "srcs": attr.label_list(allow_files = [".py"]),
    "data": attr.label_list(allow_files = True),
    "deps": attr.label_list(),
    "env": attr.label(
        mandatory = True,
        allow_files = True,
    ),
    "main": attr.label(allow_single_file = [".py"]),
    "launcher": attr.label(
        default = Label("//tools/conda_run"),
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
