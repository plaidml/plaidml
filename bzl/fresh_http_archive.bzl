def _fresh_http_archive_impl(ctx):
    ctx.download_and_extract(
        ctx.attr.url, sha256=ctx.attr.sha256, stripPrefix=ctx.attr.strip_prefix)
    args = [
        "python",
        ctx.path(ctx.attr._script),
    ]
    result = ctx.execute(args)
    if result.return_code:
        fail("clean.py failed: %s (%s)" % (result.stdout, result.stderr))
    ctx.symlink(ctx.path(ctx.attr.build_file), "BUILD")

fresh_http_archive = repository_rule(
    attrs = {
        "url": attr.string(mandatory = True),
        "sha256": attr.string(),
        "strip_prefix": attr.string(),
        "build_file": attr.label(
            allow_files = True,
            mandatory = True,
            single_file = True,
        ),
        "_script": attr.label(
            executable = True,
            default = Label("//bzl:clean.py"),
            cfg = "host",
        ),
    },
    implementation = _fresh_http_archive_impl,
)
