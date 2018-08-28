def _fresh_http_archive_impl(repository_ctx):
    url = repository_ctx.attr.url
    sha256 = repository_ctx.attr.sha256
    strip_prefix = repository_ctx.attr.strip_prefix
    repository_ctx.download_and_extract(url, sha256=sha256, stripPrefix=strip_prefix)
    args = [
        "python",
        repository_ctx.path(repository_ctx.attr._script),
    ]
    result = repository_ctx.execute(args)
    if result.return_code:
        fail("clean.py failed: %s (%s)" % (result.stdout, result.stderr))
    repository_ctx.symlink(repository_ctx.path(repository_ctx.attr.build_file), "BUILD")

fresh_http_archive = repository_rule(
    attrs = {
        "url": attr.string(mandatory = True),
        "sha256": attr.string(),
        "strip_prefix": attr.string(),
        "build_file": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "_script": attr.label(
            executable = True,
            default = Label("//bzl:clean.py"),
            cfg = "host",
        ),
    },
    implementation = _fresh_http_archive_impl,
)
