BUILD_FILE = """
package(default_visibility = ["//visibility:public"])

exports_files(["env"])
"""

def _xsmm_repo_impl(repository_ctx):
    name = repository_ctx.name
    env_path = repository_ctx.path(repository_ctx.attr.env)
    prefix_path = repository_ctx.path("env")

    args = [
        repository_ctx.which("make"),
        "-f",
        "./Makefile",
        "header-only",
    ]

    result = repository_ctx.download_and_extract(
        "https://github.com/hfp/libxsmm/archive/1.12.1.zip",
        output='.',
        sha256="451ec9d30f0890bf3081aa3d0d264942a6dea8f9d29c17bececc8465a10a832b",
        stripPrefix='libxsmm-1.12.1',
    )

    result = repository_ctx.execute(args, quiet = False, timeout = 1200)

    if result.return_code:
        fail("xmss_repo failed: %s (%s)" % (result.stdout, result.stderr))

    if repository_ctx.attr.build_file:
        repository_ctx.template("BUILD", repository_ctx.attr.build_file, {}, False)
    else:
        repository_ctx.file("BUILD", BUILD_FILE)
    
xsmm_repo = repository_rule(
    attrs = {
        "env": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "build_file": attr.label(allow_single_file = True),
    },
    
    implementation = _xsmm_repo_impl,
)
