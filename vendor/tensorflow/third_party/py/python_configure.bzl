def _impl(repository_ctx):
    repository_ctx.file("BUILD", """
toolchain(
    name = "py_toolchain",
    toolchain = "@com_intel_plaidml//:py_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)
    """)

local_python_configure = repository_rule(
    implementation = _impl,
    attrs = {
        "environ": attr.string_dict(),
        "platform_constraint": attr.string(),
    },
)

remote_python_configure = repository_rule(
    implementation = _impl,
    remotable = True,
    attrs = {
        "environ": attr.string_dict(),
        "platform_constraint": attr.string(),
    },
)

python_configure = repository_rule(
    implementation = _impl,
    attrs = {
        "platform_constraint": attr.string(),
    },
)
