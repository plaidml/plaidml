def _xsmm_repo_impl(repository_ctx):
    name = repository_ctx.name
    url = repository_ctx.attr.url
    sha256 = repository_ctx.attr.sha256
    stripPrefix = repository_ctx.attr.stripPrefix

    args = [
        repository_ctx.which("make"),
        "-f",
        "./Makefile",
        "header-only",
    ]

    result = repository_ctx.download_and_extract(
        url = url,
        sha256 = sha256,
        stripPrefix = stripPrefix,
    )

    result = repository_ctx.execute(
        args,
        quiet = False,
        timeout = 1200,
    )

    if result.return_code:
        fail("xmss_repo failed: %s (%s)" % (result.stdout, result.stderr))

    repository_ctx.template("BUILD", repository_ctx.attr.build_file, {}, False)

xsmm_repo = repository_rule(
    attrs = {
        "url": attr.string(
            mandatory = True,
        ),
        "sha256": attr.string(
            mandatory = True,
        ),
        "stripPrefix": attr.string(
            mandatory = True,
        ),
        "build_file": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
    },
    implementation = _xsmm_repo_impl,
)
