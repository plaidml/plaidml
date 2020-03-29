def is_unix(ctx):
    return ctx.host_configuration.host_path_separator == ":"

def run_python_attrs(attrs):
    result = {
        "_python": attr.label(
            default = "@com_intel_plaidml_conda_unix//:python",
            allow_single_file = True,
        ),
    }
    result.update(attrs)
    return result

def run_python_tool(ctx, args, python, tool, **kwargs):
    if is_unix(ctx):
        prefix_args = ctx.actions.args()
        prefix_args.add(tool)
        args = [prefix_args] + [args]
        tools = [python, tool]
        executable = python
    else:
        args = [args]
        tools = [tool]
        executable = tool

    ctx.actions.run(
        executable = executable,
        arguments = args,
        tools = tools,
        **kwargs
    )
