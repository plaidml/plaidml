# Tile Bazel configurations

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library", "cc_proto_library")

PY_SRCS_VER = "PY2AND3"

PLAIDML_COPTS = select({
    "@toolchain//:macos_x86_64": [
        "-std=c++14",
        "-Werror",
    ],
    "@toolchain//:windows_x86_64": [
        "/std:c++14",
        "/DWIN32_LEAN_AND_MEAN",
    ],
    "//conditions:default": [
        "-Werror",
    ],
})

PLAIDML_LINKOPTS = select({
    "//bzl:android": [
        "-Lexternal/androidndk/ndk/sources/cxx-stl/llvm-libc++/libs/armeabi-v7a/",
        "-pie",
    ],
    "@toolchain//:windows_x86_64": [],
    "@toolchain//:macos_x86_64": [],
    "//conditions:default": [
        "-pthread",
        "-lm",
    ],
})

def plaidml_cc_library(copts=[], linkopts=[], **kwargs):
    native.cc_library(copts=PLAIDML_COPTS + copts, linkopts=PLAIDML_LINKOPTS + linkopts, **kwargs)

def plaidml_objc_library(copts=[], linkopts=[], **kwargs):
    native.objc_library(copts=PLAIDML_COPTS + copts + ["-Wno-shorten-64-to-32"], **kwargs)

def plaidml_cc_binary(copts=[], **kwargs):
    native.cc_binary(copts=PLAIDML_COPTS + copts, **kwargs)

def plaidml_cc_test(copts=[], deps=(), linkopts=[], **kwargs):
    native.cc_test(
        copts=PLAIDML_COPTS + copts,
        deps=deps + [str(Label("//testing:gtest_main"))],
        linkopts=PLAIDML_LINKOPTS + linkopts,
        **kwargs)

def plaidml_py_library(**kwargs):
    native.py_library(srcs_version=PY_SRCS_VER, **kwargs)

def plaidml_py_init(name, **kwargs):
    pyinit_name = name + "_init_py"

    native.genrule(
        name=pyinit_name,
        visibility=["//visibility:private"],
        outs=["__init__.py"],
        cmd="touch $(location __init__.py)")

    plaidml_py_library(name=name, srcs=[":" + pyinit_name], **kwargs)

def plaidml_proto_library(name, **kwargs):
    plaidml_cc_proto_library(name=name, **kwargs)
    plaidml_py_proto_library(name=name, **kwargs)

def plaidml_py_proto_library(name, srcs, deps=(), srcs_version="PY2AND3", **kwargs):
    py_proto_library(
        name=name + "_py",
        srcs=srcs,
        srcs_version=srcs_version,
        deps=[d + "_py" for d in deps] + ["@com_google_protobuf//:protobuf_python"],
        protoc="@com_google_protobuf//:protoc",
        default_runtime="@com_google_protobuf//:protobuf_python",
        **kwargs)

def plaidml_cc_proto_library(name, srcs, deps=(), **kwargs):
    cc_proto_library(
        name=name + "_cc",
        srcs=srcs,
        deps=[d + "_cc" for d in deps] + ["@com_google_protobuf//:cc_wkt_protos"],
        copts=PLAIDML_COPTS,
        protoc="@com_google_protobuf//:protoc",
        default_runtime="@com_google_protobuf//:protobuf",
        **kwargs)

def plaidml_ast(name, ast, output, template="base", visibility=None):
    native.genrule(
        name=name,
        outs=[output],
        srcs=[ast],
        tools=["//base/util/astgen"],
        cmd="$(location //base/util/astgen) -i $(SRCS) -t {} -o $(OUTS)".format(template),
    )

def plaidml_grammar(name, bison_src, flex_src, outs, visibility=None):
    native.genrule(
        name=name,
        outs=outs,
        srcs=[bison_src, flex_src],
        visibility=visibility,
        cmd=select({
            "@toolchain//:macos_x86_64": """
ssrcs=($(SRCS))
/usr/local/opt/bison/bin/bison --verbose $${ssrcs[0]}
/usr/local/opt/flex/bin/flex $${ssrcs[1]}
cp %s $(@D)
""" % (" ".join(outs)),
            "@toolchain//:windows_x86_64": """
ssrcs=($(SRCS))
bison --verbose $${ssrcs[0]}
flex --nounistd $${ssrcs[1]}
cp %s $(@D)
""" % (" ".join(outs)),
            "//conditions:default": """
ssrcs=($(SRCS))
bison --verbose $${ssrcs[0]}
flex $${ssrcs[1]}
cp %s $(@D)
""" % (" ".join(outs)),
        }),
    )

def run_as_test(name, target, cmd, data=[], **kwargs):
    native.genrule(
        name=name + "_sh",
        output_to_bindir=1,
        outs=[name + ".sh"],
        cmd="echo 'exec " + ' '.join(cmd) + "' > $@")
    data = data + [target]
    native.sh_test(name=name, srcs=[":" + name + "_sh"], data=data, **kwargs)

def plaidml_cp(name, files):
    native.genrule(
        name=name,
        srcs=[files[k] for k in files],
        outs=[k for k in files],
        cmd="; ".join(["cp $(location %s) $(location %s)" % (files[k], k) for k in files]))

def _plaidml_py_wheel_impl(ctx):
    tpl = ctx.file._setup_py_tpl
    setup_py = ctx.new_file(ctx.label.name + '.pkg/setup.py')
    pkg_inputs = depset([setup_py])
    version = ctx.var.get('version', default='unknown')
    if ctx.file.config:
        cfg = ctx.new_file(setup_py, 'setup.cfg')
        ctx.actions.run_shell(
            outputs=[cfg],
            inputs=[ctx.file.config],
            command="cp $1 $2",
            arguments=[ctx.file.config.path, cfg.path],
            mnemonic="CopySetupCfg")
        pkg_inputs += [cfg]
    build_src_base = ctx.build_file_path.rsplit('/', 1)[0] + "/"
    pkg_prefix = ctx.attr.package_prefix
    if pkg_prefix != '':
        pkg_prefix = '/' + pkg_prefix
    for tgt in ctx.attr.srcs:
        for src in tgt.files + tgt.data_runfiles.files:
            dest = ctx.new_file(setup_py, 'pkg' + pkg_prefix + src.path[src.path.find(build_src_base) + len(build_src_base) - 1:])
            print("src.path: {0}\ndest.path: {1}\nsrc.basename: {2}\nbuild_src_base: {3}\n".format(src.path, dest.path, src.basename, build_src_base))
            ctx.actions.run_shell(
                outputs=[dest],
                inputs=[src],
                command="cp $1 $2",
                arguments=[src.path, dest.path],
                mnemonic="CopyPackageFile")
            pkg_inputs += [dest]
    ctx.actions.expand_template(
        template=tpl,
        output=setup_py,
        substitutions={
            'bzl_package_name': ctx.attr.package_name,
            'bzl_version': version,
            'bzl_target_cpu': ctx.var['TARGET_CPU']
        })
    wheel_filename = "dist/%s-%s-%s-%s-%s.whl" % (ctx.attr.package_name, version, ctx.attr.python, ctx.attr.abi,
                                                  ctx.attr.platform)
    wheel = ctx.new_file(setup_py, wheel_filename)
    bdist_wheel_args = [setup_py.path, "--no-user-cfg", "bdist_wheel"]
    if ctx.attr.platform != 'any':
        bdist_wheel_args.append("--plat-name")
        bdist_wheel_args.append(ctx.attr.platform)
    ctx.actions.run(
        outputs=[wheel],
        inputs=pkg_inputs.to_list(),
        executable="python",
        arguments=bdist_wheel_args,
        mnemonic="BuildWheel",
        use_default_shell_env=True,
    )
    output = ctx.new_file(ctx.bin_dir, wheel.basename)
    ctx.actions.run_shell(
        outputs=[output],
        inputs=[wheel],
        command="cp $1 $2",
        arguments=[wheel.path, output.path],
        mnemonic="CopyWheel")
    return DefaultInfo(files=depset([output]))

plaidml_py_wheel = rule(
    attrs = {
        "srcs": attr.label_list(
            mandatory = True,
            allow_files = True,
        ),
        "config": attr.label(allow_single_file = [".cfg"]),
        "package_name": attr.string(mandatory = True),
        "package_prefix": attr.string(default = ""),
        "python": attr.string(mandatory = True),
        "abi": attr.string(default = "none"),
        "platform": attr.string(default = "any"),
        "_setup_py_tpl": attr.label(
            default = Label("//bzl:setup.tpl.py"),
            allow_single_file = True,
        ),
    },
    implementation = _plaidml_py_wheel_impl,
)

def _plaidml_version_impl(ctx):
    ctx.actions.expand_template(
        template=ctx.file._template,
        output=ctx.outputs.version_file,
        substitutions={
            "{PREFIX}": ctx.attr.prefix,
            "{VERSION}": ctx.var.get('version', default='unknown'),
        })

plaidml_cc_version = rule(
    attrs = {
        "prefix": attr.string(mandatory = True),
        "_template": attr.label(
            default = Label("//bzl:version.tpl.cc"),
            allow_files = True,
            single_file = True,
        ),
    },
    output_to_genfiles = True,
    outputs = {"version_file": "_version.cc"},
    implementation = _plaidml_version_impl,
)

def _venv_wrapper_impl(ctx):
    main = ctx.expand_location("$(location {})".format(ctx.attr.main.label), [ctx.attr.main])
    requirements = ctx.expand_location("$(location {})".format(ctx.attr.requirements.label),
                                       [ctx.attr.requirements])
    venv_args = str(ctx.attr.venv_args)
    ctx.actions.expand_template(
        template=ctx.file._template,
        output=ctx.outputs.executable,
        substitutions={
            "__BZL_MAIN__": main,
            "__BZL_REQUIREMENTS__": requirements,
            "__BZL_VENV_ARGS__": venv_args,
            "__BZL_WORKSPACE__": ctx.workspace_name,
        })

venv_wrapper = rule(
    attrs = {
        "main": attr.label(
            allow_files = True,
            mandatory = True,
            single_file = True,
        ),
        "requirements": attr.label(
            allow_files = True,
            mandatory = True,
            single_file = True,
        ),
        "venv_args": attr.string_list(default = []),
        "_template": attr.label(
            default = Label("//bzl:venv.tpl.py"),
            allow_files = True,
            single_file = True,
        ),
    },
    executable = True,
    implementation = _venv_wrapper_impl,
)

def plaidml_py_binary(name,
                      main=None,
                      srcs=[],
                      requirements="requirements.txt",
                      data=[],
                      venv_args=[],
                      **kwargs):
    if main == None:
        main = name + ".py"
    venv = name + "__venv__.py"
    venv_wrapper(
        name=venv,
        main=main,
        requirements=requirements,
        venv_args=venv_args,
    )
    native.py_binary(
        name=name,
        main=venv,
        srcs=srcs + [venv],
        srcs_version=PY_SRCS_VER,
        data=data + [requirements],
        **kwargs)

def plaidml_py_test(name,
                    main=None,
                    srcs=[],
                    requirements="requirements.txt",
                    data=[],
                    venv_args=[],
                    **kwargs):
    if main == None:
        main = name + ".py"
    venv = name + "__venv__.py"
    venv_wrapper(
        name=venv,
        main=main,
        requirements=requirements,
        venv_args=venv_args,
    )
    native.py_test(
        name=name,
        main=venv,
        srcs=srcs + [venv],
        srcs_version=PY_SRCS_VER,
        data=data + [requirements],
        **kwargs)
