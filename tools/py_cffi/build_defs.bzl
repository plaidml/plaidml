load("@rules_python//python:defs.bzl", "py_library")

def _py_cffi_impl(ctx):
    args = ctx.actions.args()
    args.add_all(ctx.files.srcs, before_each = "--source")
    args.add("--module", ctx.attr.module)
    args.add("--output", ctx.outputs.out)
    ctx.actions.run(
        inputs = ctx.files.srcs,
        outputs = [ctx.outputs.out],
        arguments = [args],
        tools = [ctx.executable._tool],
        executable = ctx.executable._tool,
        mnemonic = "PyCffi",
        use_default_shell_env = True,
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

py_cffi_rule = rule(
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "module": attr.string(
            mandatory = True,
        ),
        "out": attr.output(),
        "_tool": attr.label(
            default = Label("//tools/py_cffi"),
            executable = True,
            cfg = "host",
        ),
    },
    implementation = _py_cffi_impl,
)

# It's named srcs_ordered because we want to prevent buildifier from automatically sorting this list.
def py_cffi(name, module, srcs_ordered, **kwargs):
    out = name + ".py"
    py_cffi_rule(
        name = name + "_py_cffi",
        module = module,
        srcs = srcs_ordered,
        out = out,
    )

    py_library(
        name = name,
        srcs = [out],
        **kwargs
    )
