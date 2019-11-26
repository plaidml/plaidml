def _xsmm_repo_impl(repository_ctx):
    make = repository_ctx.which("make")
    if not make:
        fail("xsmm_repo failed: 'make' could not be found. " +
             "If you are on Windows, did you run 'conda activate .cenv\\'?")

    repository_ctx.download_and_extract(
        url = repository_ctx.attr.url,
        sha256 = repository_ctx.attr.sha256,
        stripPrefix = repository_ctx.attr.strip_prefix,
    )

    result = repository_ctx.execute(
        [make, "-f", "Makefile", "header-only"],
        quiet = False,
    )
    if result.return_code:
        fail("xsmm_repo failed: %s (%s)" % (result.stdout, result.stderr))

    repository_ctx.template("BUILD", repository_ctx.attr.build_file, {}, False)

xsmm_repo = repository_rule(
    attrs = {
        "url": attr.string(mandatory = True),
        "sha256": attr.string(),
        "strip_prefix": attr.string(),
        "build_file": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
    },
    implementation = _xsmm_repo_impl,
)
