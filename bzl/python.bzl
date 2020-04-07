def is_unix(ctx):
    return ctx.host_configuration.host_path_separator == ":"

# This rule is to ensure that python-based build rules only use the python
# that is bundled with the internal conda environment to maintain hermicity.
# If we attempt to run the `tool` directly, then the system installed python
# is used (due to the tool's entrypoint saying "#!/usr/bin/env python").
# However, on Windows, the tool is an actual `.exe`, which means that we don't
# need to run python directly, instead we can run the `tool` directly.
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
