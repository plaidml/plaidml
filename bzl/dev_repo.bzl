def workspace_and_buildfile(ctx):
    """Utility function for writing WORKSPACE and, if requested, a BUILD file.

    This rule is inteded to be used in the implementation function of a
    repository rule.
    It assumes the parameters `name`, `build_file`, `build_file_contents`,
    `workspace_file`, and `workspace_file_content` to be
    present in `ctx.attr`, the latter four possibly with value None.

    Args:
      ctx: The repository context of the repository rule calling this utility
        function.
    """
    if ctx.attr.build_file and ctx.attr.build_file_content:
        ctx.fail("Only one of build_file and build_file_content can be provided.")

    if ctx.attr.workspace_file and ctx.attr.workspace_file_content:
        ctx.fail("Only one of workspace_file and workspace_file_content can be provided.")

    if ctx.attr.workspace_file:
        ctx.delete("WORKSPACE")
        ctx.symlink(ctx.attr.workspace_file, "WORKSPACE")
    elif ctx.attr.workspace_file_content:
        ctx.delete("WORKSPACE")
        ctx.file("WORKSPACE", ctx.attr.workspace_file_content)
    else:
        ctx.file("WORKSPACE", "workspace(name = \"{name}\")\n".format(name = ctx.name))

    if ctx.attr.build_file:
        ctx.delete("BUILD.bazel")
        ctx.symlink(ctx.attr.build_file, "BUILD.bazel")
    elif ctx.attr.build_file_content:
        ctx.delete("BUILD.bazel")
        ctx.file("BUILD.bazel", ctx.attr.build_file_content)

def update_attrs(orig, keys, override):
    """Utility function for altering and adding the specified attributes to a particular repository rule invocation.

     This is used to make a rule reproducible.

    Args:
        orig: dict of actually set attributes (either explicitly or implicitly)
            by a particular rule invocation
        keys: complete set of attributes defined on this rule
        override: dict of attributes to override or add to orig

    Returns:
        dict of attributes with the keys from override inserted/updated
    """
    result = {}
    for key in keys:
        if getattr(orig, key) != None:
            result[key] = getattr(orig, key)
    result["name"] = orig.name
    result.update(override)
    return result

def _dev_http_archive_impl(ctx):
    """Implementation of the http_archive rule."""
    if not ctx.attr.url and not ctx.attr.urls:
        fail("At least one of url and urls must be provided")
    if ctx.attr.build_file and ctx.attr.build_file_content:
        fail("Only one of build_file and build_file_content can be provided.")

    all_urls = []
    if ctx.attr.urls:
        all_urls = ctx.attr.urls
    if ctx.attr.url:
        all_urls = [ctx.attr.url] + all_urls

    env_key = "BAZEL_" + ctx.name.upper() + "_PATH"
    if env_key in ctx.os.environ:
        repo_path = ctx.os.environ[env_key]
        script_path = ctx.path(Label("@com_intel_plaidml//bzl:dev_repo.py"))

        result = ctx.execute([
            ctx.which("python"),
            script_path,
            repo_path,
            ctx.path("."),
        ], quiet = False)

        if not result.return_code == 0:
            fail("dev_http_archive failure: %s\n" % result.stderr)

        workspace_and_buildfile(ctx)

        return update_attrs(ctx.attr, _dev_http_archive_attrs.keys(), {})

    download_info = ctx.download_and_extract(
        all_urls,
        "",
        ctx.attr.sha256,
        ctx.attr.type,
        ctx.attr.strip_prefix,
    )

    workspace_and_buildfile(ctx)

    return update_attrs(ctx.attr, _dev_http_archive_attrs.keys(), {"sha256": download_info.sha256})

_dev_http_archive_attrs = {
    "url": attr.string(),
    "urls": attr.string_list(),
    "sha256": attr.string(),
    "strip_prefix": attr.string(),
    "type": attr.string(),
    "build_file": attr.label(allow_single_file = True),
    "build_file_content": attr.string(),
    "workspace_file": attr.label(allow_single_file = True),
    "workspace_file_content": attr.string(),
}

dev_http_archive = repository_rule(
    implementation = _dev_http_archive_impl,
    attrs = _dev_http_archive_attrs,
    local = True,
)
