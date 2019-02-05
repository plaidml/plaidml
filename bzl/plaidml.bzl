# Tile Bazel configurations

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")

PY_SRCS_VER = "PY2AND3"

PLAIDML_COPTS = select({
    "@toolchain//:windows_x86_64": [
        "/std:c++14",
        "/DWIN32_LEAN_AND_MEAN",
    ],
    "//conditions:default": [
        "-std=c++14",
        "-Werror",
    ],
})

PLAIDML_LINKOPTS = select({
    "@toolchain//:windows_x86_64": [],
    "@toolchain//:macos_x86_64": [],
    "//conditions:default": [
        "-pthread",
        "-lm",
    ],
})

def plaidml_cc_library(copts = [], linkopts = [], **kwargs):
    native.cc_library(copts = PLAIDML_COPTS + copts, linkopts = PLAIDML_LINKOPTS + linkopts, **kwargs)

def plaidml_objc_library(copts = [], linkopts = [], **kwargs):
    native.objc_library(copts = PLAIDML_COPTS + copts + ["-Wno-shorten-64-to-32"], **kwargs)

def plaidml_cc_binary(copts = [], **kwargs):
    native.cc_binary(copts = PLAIDML_COPTS + copts, **kwargs)

def plaidml_cc_test(copts = [], deps = (), linkopts = [], **kwargs):
    native.cc_test(
        copts = PLAIDML_COPTS + copts,
        deps = deps + [str(Label("//testing:gtest_main"))],
        linkopts = PLAIDML_LINKOPTS + linkopts,
        **kwargs
    )

def plaidml_py_library(**kwargs):
    native.py_library(srcs_version = PY_SRCS_VER, **kwargs)

def plaidml_py_init(name, **kwargs):
    pyinit_name = name + "_init_py"

    native.genrule(
        name = pyinit_name,
        visibility = ["//visibility:private"],
        outs = ["__init__.py"],
        cmd = "touch $(location __init__.py)",
    )

    plaidml_py_library(name = name, srcs = [":" + pyinit_name], **kwargs)

def plaidml_proto_library(name, **kwargs):
    plaidml_cc_proto_library(name = name, **kwargs)
    plaidml_py_proto_library(name = name, **kwargs)

def plaidml_py_proto_library(name, srcs, deps = (), srcs_version = "PY2AND3", **kwargs):
    py_proto_library(
        name = name + "_py",
        srcs = srcs,
        srcs_version = srcs_version,
        deps = [d + "_py" for d in deps] + ["@com_google_protobuf//:protobuf_python"],
        protoc = "@com_google_protobuf//:protoc",
        default_runtime = "@com_google_protobuf//:protobuf_python",
        **kwargs
    )

def plaidml_cc_proto_library(name, srcs, deps = (), **kwargs):
    cc_proto_library(
        name = name + "_cc",
        srcs = srcs,
        deps = [d + "_cc" for d in deps] + ["@com_google_protobuf//:cc_wkt_protos"],
        copts = PLAIDML_COPTS,
        protoc = "@com_google_protobuf//:protoc",
        default_runtime = "@com_google_protobuf//:protobuf",
        **kwargs
    )

def plaidml_ast(name, ast, output, template = "base", visibility = None):
    native.genrule(
        name = name,
        outs = [output],
        srcs = [ast],
        tools = ["//base/util/astgen"],
        cmd = "$(location //base/util/astgen) -i $(SRCS) -t {} -o $(OUTS)".format(template),
    )

def plaidml_bison(name, src, out, defines, visibility = None):
    COMMON_ARGS = "-o $(location %s) --defines=$(location %s) $(SRCS)" % (out, defines)
    native.genrule(
        name = name,
        srcs = [src],
        outs = [out, defines],
        visibility = visibility,
        cmd = "bison --verbose " + COMMON_ARGS,
    )

def plaidml_flex(name, src, out, hdr, visibility = None):
    COMMON_ARGS = "-o $(location %s) --header-file=$(location %s) $(SRCS)" % (out, hdr)
    native.genrule(
        name = name,
        srcs = [src],
        outs = [out, hdr],
        visibility = visibility,
        cmd = select({
            "@toolchain//:windows_x86_64": "flex --nounistd " + COMMON_ARGS,
            "//conditions:default": "flex " + COMMON_ARGS,
        }),
    )

def plaidml_cp(name, files):
    native.genrule(
        name = name,
        srcs = [files[k] for k in files],
        outs = [k for k in files],
        cmd = "; ".join(["cp $(location %s) $(location %s)" % (files[k], k) for k in files]),
    )

def _plaidml_py_wheel_impl(ctx):
    tpl = ctx.file._setup_py_tpl
    setup_py = ctx.new_file(ctx.label.name + ".pkg/setup.py")
    wheel_inputs = depset([setup_py])
    version = ctx.var.get("version", default = "unknown")
    if ctx.file.config:
        cfg = ctx.new_file(setup_py, "setup.cfg")
        ctx.actions.expand_template(
            template = ctx.file.config,
            output = cfg,
            substitutions = ctx.attr.config_substitutions,
        )
        wheel_inputs += [cfg]
    build_src_base = ctx.build_file_path.rsplit("/", 1)[0] + "/"
    pkg_prefix = ctx.attr.package_prefix
    if pkg_prefix != "":
        pkg_prefix = "/" + pkg_prefix
    for tgt in ctx.attr.srcs:
        for src in tgt.files + tgt.data_runfiles.files:
            dest = ctx.new_file(setup_py, "pkg" + pkg_prefix + src.path[src.path.find(build_src_base) + len(build_src_base) - 1:])
            ctx.actions.run_shell(
                outputs = [dest],
                inputs = [src],
                command = "cp $1 $2",
                arguments = [src.path, dest.path],
                mnemonic = "CopyPackageFile",
            )
            wheel_inputs += [dest]
    for tgt in ctx.attr.data:
        for src in tgt.files + tgt.data_runfiles.files:
            basename = src.basename
            if basename in ctx.attr.data_renames:
                basename = ctx.attr.data_renames[basename]
            dest = ctx.new_file(setup_py, "data/" + basename)
            ctx.actions.run_shell(
                outputs = [dest],
                inputs = [src],
                command = "cp $1 $2",
                arguments = [src.path, dest.path],
                mnemonic = "CopyDataFile",
            )
            wheel_inputs += [dest]
    ctx.actions.expand_template(
        template = tpl,
        output = setup_py,
        substitutions = {
            "bzl_package_name": ctx.attr.package_name,
            "bzl_version": version,
            "bzl_target_cpu": ctx.var["TARGET_CPU"],
        },
    )
    wheel_filename = "dist/%s-%s-%s-%s-%s.whl" % (
        ctx.attr.package_name,
        version,
        ctx.attr.python,
        ctx.attr.abi,
        ctx.attr.platform,
    )
    wheel = ctx.new_file(setup_py, wheel_filename)
    bdist_wheel_args = [setup_py.path, "--no-user-cfg", "bdist_wheel"]
    if ctx.attr.platform != "any":
        bdist_wheel_args.append("--plat-name")
        bdist_wheel_args.append(ctx.attr.platform)
    ctx.actions.run(
        outputs = [wheel],
        inputs = wheel_inputs.to_list(),
        executable = "python",
        arguments = bdist_wheel_args,
        mnemonic = "BuildWheel",
        use_default_shell_env = True,
    )
    output = ctx.new_file(ctx.bin_dir, wheel.basename)
    ctx.actions.run_shell(
        outputs = [output],
        inputs = [wheel],
        command = "cp $1 $2",
        arguments = [wheel.path, output.path],
        mnemonic = "CopyWheel",
    )
    return DefaultInfo(files = depset([output]))

plaidml_py_wheel = rule(
    attrs = {
        "srcs": attr.label_list(
            mandatory = True,
            allow_files = True,
        ),
        "data": attr.label_list(
            allow_empty = True,
            allow_files = True,
        ),
        "data_renames": attr.string_dict(),
        "config": attr.label(allow_single_file = [".cfg"]),
        "config_substitutions": attr.string_dict(),
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
        template = ctx.file._template,
        output = ctx.outputs.version_file,
        substitutions = {
            "{PREFIX}": ctx.attr.prefix,
            "{VERSION}": ctx.var.get("version", default = "unknown"),
        },
    )

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

def plaidml_macos_dylib(name, lib, src, tags, internal_libname = ""):
    # Builds an output .dylib with the runtime path set to @rpath/{name}.
    # The output is a single file, ${lib}, which should end with ".dylib".
    if not internal_libname:
        internal_libname = lib
    native.genrule(
        name = name,
        tags = tags,
        srcs = [src],
        outs = [lib],
        message = "Setting rpath for " + lib,
        cmd = "lib=\"" + lib + "\"; internal_libname=\"" + internal_libname + "\"" + """
            cp $< $@
            original_mode=$$(stat -f%#p $@)
            chmod u+w $@
            install_name_tool -id @rpath/$${internal_libname} $@
            chmod $${original_mode} $@
        """,
    )
