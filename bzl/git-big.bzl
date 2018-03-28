def _git_big_repo_impl(ctx):
    ctx.file("WORKSPACE", "")
    ctx.file("BUILD", """
exports_files(["env"])
""")
    ctx.file("env", ctx.attr.workspace_dir)

git_big_repo = repository_rule(
    attrs = {
        "workspace_dir": attr.string(mandatory = True),
    },
    implementation = _git_big_repo_impl,
)

def _git_big_impl(ctx):
    out = ctx.outputs.out
    args = [
        ctx.file.env.path,
        ctx.attr.src,
        out.path,
    ]
    ctx.actions.run_shell(
        outputs = [out],
        arguments = args,
        use_default_shell_env = True,
        command = "cd $(cat $1); git big pull --hard $2 --extra $3",
    )
    return DefaultInfo(files=depset([out]))

git_big_rule = rule(
    attrs = {
        "src": attr.string(),
        "out": attr.output(mandatory = True),
        "env": attr.label(
            allow_files = True,
            single_file = True,
            default = Label("@git_big//:env"),
        ),
    },
    implementation = _git_big_impl,
)

def git_big(name, src=None, **kwargs):
    if src == None:
        reldir, name = name.split(':')
        src = '/'.join([reldir, name])
    rule_name = name + ".gitbig"
    git_big_rule(
        name = rule_name,
        src = src,
        out = name,
        tags = ["git_big"],
    )
    return rule_name
