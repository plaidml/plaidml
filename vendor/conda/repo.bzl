BUILD_FILE = """
package(default_visibility = ["//visibility:public"])

exports_files(["env"])
"""

def _conda_repo_impl(repository_ctx):
    if "windows" in repository_ctx.os.name:
        env = repository_ctx.attr.env_windows
        build_file = repository_ctx.attr.build_file_windows
    else:
        env = repository_ctx.attr.env_unix
        build_file = repository_ctx.attr.build_file_unix

    name = repository_ctx.name
    wrapper = repository_ctx.path(repository_ctx.attr._wrapper)

    env_path = repository_ctx.path(env)
    prefix_path = repository_ctx.path("env")
    args = [
        repository_ctx.which("python"),
        wrapper,
        "env",
        "create",
        "-f",
        env_path,
        "-p",
        prefix_path,
    ]
    result = repository_ctx.execute(args, quiet = False, timeout = 1800)
    if result.return_code:
        fail("conda_repo failed: %s (%s)" % (result.stdout, result.stderr))

    if build_file:
        repository_ctx.template("BUILD", build_file, {}, False)
    else:
        repository_ctx.file("BUILD", BUILD_FILE)

conda_repo = repository_rule(
    attrs = {
        "env_windows": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "env_unix": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "build_file_windows": attr.label(allow_single_file = True),
        "build_file_unix": attr.label(allow_single_file = True),
        "_wrapper": attr.label(
            default = Label("//vendor/conda:wrapper.py"),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
    },
    implementation = _conda_repo_impl,
)
