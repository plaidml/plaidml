load("@bazel_skylib//lib:paths.bzl", "paths")
load("//bzl:python.bzl", "run_python_tool")

def _py_setup_impl(ctx):
    version = ctx.var.get("version", default = "0.0.0")
    wheel_filename = "%s-%s-%s-%s-%s.whl" % (
        ctx.attr.package_name,
        version,
        ctx.attr.python_version,
        ctx.attr.abi,
        ctx.attr.platform,
    )
    pkg_path = ctx.label.name + ".tmp"
    dist_path = paths.join(pkg_path, "tmp", "dist")
    wheel_path = paths.join(dist_path, wheel_filename)
    pkg_dir = ctx.actions.declare_directory(pkg_path)
    wheel = ctx.actions.declare_file(wheel_path)

    args = ctx.actions.args()
    args.add("--no-user-cfg")
    args.add(ctx.attr.action)
    if ctx.attr.universal:
        args.add("--universal")
    if ctx.attr.platform != "any":
        args.add("--plat-name", ctx.attr.platform)

    run_python_tool(
        ctx,
        mnemonic = "PySetup",
        python = ctx.file.python,
        tool = ctx.executable.tool,
        args = args,
        outputs = [pkg_dir, wheel],
        env = {
            "BZL_SRC": ctx.executable.tool.path,
            "BZL_TGT": pkg_dir.path,
            "BZL_WORKSPACE": ctx.workspace_name,
            "BZL_VERSION": version,
        },
    )

    return DefaultInfo(files = depset([wheel]))

py_setup = rule(
    attrs = {
        "python": attr.label(
            default = "@com_intel_plaidml_conda//:python",
            allow_single_file = True,
        ),
        "tool": attr.label(
            mandatory = True,
            executable = True,
            cfg = "host",
        ),
        "universal": attr.bool(),
        "abi": attr.string(default = "none"),
        "action": attr.string(default = "bdist_wheel"),
        "package_name": attr.string(mandatory = True),
        "platform": attr.string(default = "any"),
        "python_version": attr.string(default = "py2.py3"),
    },
    implementation = _py_setup_impl,
)
