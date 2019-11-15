# Tile Bazel configurations

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")
load("@rules_python//python:defs.bzl", "py_library")

PY_SRCS_VER = "PY2AND3"

PLAIDML_COPTS = select({
    "@com_intel_plaidml//toolchain:windows_x86_64": [
        "/std:c++17",  # This MUST match all other compilation units
        "/Zc:__cplusplus",
        "/Zc:inline",
        "/Zc:strictStrings",
        "/DWIN32_LEAN_AND_MEAN",
    ],
    "//conditions:default": [
        "-std=c++17",
        "-Werror",
    ],
})

PLAIDML_LINKOPTS = select({
    "@com_intel_plaidml//toolchain:windows_x86_64": [],
    "@com_intel_plaidml//toolchain:macos_x86_64": [],
    "//conditions:default": [
        "-pthread",
        "-lm",
    ],
})

PLATFORM_TAGS = {
    "@com_intel_plaidml//toolchain:windows_x86_64": ["msvc"],
    "@com_intel_plaidml//toolchain:macos_x86_64": ["darwin"],
    "//conditions:default": ["linux"],
}

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
        linkstatic = select({
            "@com_intel_plaidml//toolchain:linux_x86_64": 1,
            "//conditions:default": None,
        }),
        linkopts = PLAIDML_LINKOPTS + linkopts,
        **kwargs
    )

def plaidml_py_library(**kwargs):
    py_library(srcs_version = PY_SRCS_VER, **kwargs)

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

def _plaidml_bison_impl(ctx):
    args = ctx.actions.args()
    args.add("-o", ctx.outputs.out)
    args.add("--defines={}".format(ctx.outputs.defines.path))
    args.add("--verbose")
    args.add_all(ctx.files.src)
    outputs = [ctx.outputs.out, ctx.outputs.defines]
    ctx.actions.run(
        inputs = ctx.files.src,
        outputs = outputs,
        arguments = [args],
        env = ctx.attr.env,
        tools = [ctx.executable.tool],
        executable = ctx.executable.tool,
        mnemonic = "bison",
    )
    return [DefaultInfo(files = depset(outputs))]

plaidml_bison_rule = rule(
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "env": attr.string_dict(),
        "tool": attr.label(
            mandatory = True,
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
    },
    output_to_genfiles = True,
    outputs = {
        "out": "%{name}.y.cc",
        "defines": "%{name}.y.h",
    },
    implementation = _plaidml_bison_impl,
)

def plaidml_bison(name, src):
    plaidml_bison_rule(
        name = name,
        src = src,
        env = select({
            "//toolchain:windows_x86_64": {},
            "//conditions:default": {"PATH": "external/com_intel_plaidml_conda_unix/env/bin"},
        }),
        tool = select({
            "//toolchain:windows_x86_64": "@com_intel_plaidml_conda_windows//:bison",
            "//conditions:default": "@com_intel_plaidml_conda_unix//:bison",
        }),
    )

def _plaidml_flex_impl(ctx):
    args = ctx.actions.args()
    args.add("-o", ctx.outputs.out)
    args.add("--header-file={}".format(ctx.outputs.hdr.path))
    args.add_all(ctx.attr.flags)
    args.add_all(ctx.files.src)
    outputs = [ctx.outputs.out, ctx.outputs.hdr]
    ctx.actions.run(
        inputs = ctx.files.src,
        outputs = outputs,
        arguments = [args],
        use_default_shell_env = True,
        tools = [ctx.executable.tool],
        executable = ctx.executable.tool,
        mnemonic = "flex",
    )
    return [DefaultInfo(files = depset(outputs))]

plaidml_flex_rule = rule(
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "flags": attr.string_list(),
        "tool": attr.label(
            mandatory = True,
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
    },
    output_to_genfiles = True,
    outputs = {
        "out": "%{name}.cc",
        "hdr": "%{name}.h",
    },
    implementation = _plaidml_flex_impl,
)

def plaidml_flex(name, src):
    plaidml_flex_rule(
        name = name,
        src = src,
        flags = select({
            "//toolchain:windows_x86_64": ["--nounistd"],
            "//conditions:default": [],
        }),
        tool = select({
            "//toolchain:windows_x86_64": "@com_intel_plaidml_conda_windows//:flex",
            "//conditions:default": "@com_intel_plaidml_conda_unix//:flex",
        }),
    )

def _plaidml_py_wheel_impl(ctx):
    tpl = ctx.file._setup_py_tpl
    setup_py = ctx.actions.declare_file(ctx.label.name + ".pkg/setup.py")
    wheel_inputs = [setup_py]
    version = ctx.var.get("version", default = "unknown")
    if ctx.file.config:
        cfg = ctx.actions.declare_file("setup.cfg", sibling = setup_py)
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
        for src in tgt.files.to_list() + tgt.data_runfiles.files.to_list():
            dest = ctx.actions.declare_file("pkg" + pkg_prefix + src.path[src.path.find(build_src_base) + len(build_src_base) - 1:], sibling = setup_py)
            ctx.actions.run_shell(
                outputs = [dest],
                inputs = [src],
                command = "cp $1 $2",
                arguments = [src.path, dest.path],
                mnemonic = "CopyPackageFile",
            )
            wheel_inputs += [dest]
    for tgt in ctx.attr.data:
        for src in tgt.files.to_list() + tgt.data_runfiles.files.to_list():
            basename = src.basename
            if basename in ctx.attr.data_renames:
                basename = ctx.attr.data_renames[basename]
            dest = ctx.actions.declare_file("data/" + basename, sibling = setup_py)
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
    wheel = ctx.actions.declare_file(wheel_filename, sibling = setup_py)
    bdist_wheel_args = [setup_py.path, "--no-user-cfg", "bdist_wheel"]
    if ctx.attr.platform != "any":
        bdist_wheel_args.append("--plat-name")
        bdist_wheel_args.append(ctx.attr.platform)
    ctx.actions.run(
        outputs = [wheel],
        inputs = wheel_inputs,
        tools = [ctx.executable.tool],
        executable = ctx.executable.tool,
        arguments = bdist_wheel_args,
        mnemonic = "BuildWheel",
        use_default_shell_env = True,
    )
    runfiles = ctx.runfiles(files = [wheel])
    return DefaultInfo(files = depset([wheel]), runfiles = runfiles)

plaidml_py_wheel_rule = rule(
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
        "tool": attr.label(
            mandatory = True,
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
    },
    implementation = _plaidml_py_wheel_impl,
)

def plaidml_py_wheel(
        name,
        config,
        srcs,
        package_name,
        python,
        data = [],
        data_renames = {},
        config_substitutions = {},
        package_prefix = "",
        abi = "none",
        platform = "any",
        **kwargs):
    plaidml_py_wheel_rule(
        name = name,
        srcs = srcs,
        data = data,
        data_renames = data_renames,
        config = config,
        config_substitutions = config_substitutions,
        package_name = package_name,
        package_prefix = package_prefix,
        python = python,
        abi = abi,
        platform = platform,
        tool = select({
            "//toolchain:windows_x86_64": "@com_intel_plaidml_conda_windows//:python",
            "//conditions:default": "@com_intel_plaidml_conda_unix//:python",
        }),
        **kwargs
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
            allow_single_file = True,
        ),
    },
    output_to_genfiles = True,
    outputs = {"version_file": "_version.cc"},
    implementation = _plaidml_version_impl,
)

plaidml_py_version = rule(
    attrs = {
        "prefix": attr.string(mandatory = True),
        "_template": attr.label(
            default = Label("//bzl:version.tpl.py"),
            allow_single_file = True,
        ),
    },
    output_to_genfiles = True,
    outputs = {"version_file": "_version.py"},
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

def _shlib_name_patterns(name):
    return {
        "@com_intel_plaidml//toolchain:windows_x86_64": ["{}.dll".format(name)],
        "@com_intel_plaidml//toolchain:macos_x86_64": ["lib{}.dylib".format(name)],
        "//conditions:default": ["lib{}.so".format(name)],
    }

def plaidml_cc_shlib(
        name,
        shlib_name = None,
        copts = [],
        linkopts = [],
        visibility = None,
        **kwargs):
    if shlib_name == None:
        shlib_name = name
    names = _shlib_name_patterns(shlib_name)
    for key, name_list in names.items():
        for name_os in name_list:
            native.cc_binary(
                name = name_os,
                copts = PLAIDML_COPTS + copts,
                linkopts = PLAIDML_LINKOPTS + linkopts,
                linkshared = 1,
                tags = PLATFORM_TAGS[key],
                visibility = visibility,
                **kwargs
            )
    native.filegroup(
        name = name,
        srcs = select(names),
        visibility = visibility,
    )
