def _tpl(ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl.replace(":", "/")
    ctx.template(
        out,
        Label("@com_intel_plaidml//vendor/cm:%s.tpl" % tpl),
        substitutions,
    )

def _create_cm_dummy_repository(ctx):
    _tpl(ctx, "build_defs.bzl", {
        "%{cm_is_configured}": "False",
    })

    genrules = [
    ]

    _tpl(ctx, "BUILD", {
        "%{cm_include_genrules}": "\n".join(genrules),
        "%{cm_headers}": '":cm-include",',
    })

def _create_cm_repository(ctx):
    _tpl(ctx, "build_defs.bzl", {
        "%{cm_is_configured}": "True",
    })

    genrules = [
    ]

    _tpl(ctx, "BUILD", {
        "%{cm_include_genrules}": "\n".join(genrules),
        "%{cm_headers}": '":cm-include",',
    })

def _configure_cm_impl(ctx):
    _VAI_NEED_CM = ctx.os.environ.get("VAI_NEED_CM", "0").strip()

    if _VAI_NEED_CM == "1":
        _create_cm_repository(ctx)
    else:
        _create_cm_dummy_repository(ctx)

configure_cm = repository_rule(
    environ = [
    ],
    implementation = _configure_cm_impl,
)
