def _conda_repo_impl(repository_ctx):
    name = repository_ctx.name
    wrapper = repository_ctx.path(repository_ctx.attr._wrapper)

    envs = []
    for key, value in repository_ctx.attr.specs.items():
        envs.append(key)
        spec_path = repository_ctx.path(Label(value))
        prefix_path = repository_ctx.path(key)
        args = [
            "python",
            wrapper,
            "env",
            "create",
            "-f",
            spec_path,
            "-p",
            prefix_path,
        ]
        result = repository_ctx.execute(args, quiet = False, timeout=1200)
        if result.return_code:
            fail("conda_repo failed: %s (%s)" % (result.stdout, result.stderr))

    repository_ctx.file("BUILD", """
exports_files({})
""".format(envs))

conda_repo = repository_rule(
    attrs = {
        "specs": attr.string_dict(mandatory = True),
        "_wrapper": attr.label(
            default = Label("//bzl:conda_wrapper.py"),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
    },
    implementation = _conda_repo_impl,
)
