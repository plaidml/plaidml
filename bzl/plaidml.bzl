# Copyright 2020 Intel Corporation.

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

PLAIDML_COPTS = select({
    "@com_intel_plaidml//:msvc": [
        "/std:c++17",  # This MUST match all other compilation units
        "/wd4624",
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
    "@bazel_tools//src/conditions:windows": [],
    "@bazel_tools//src/conditions:darwin_x86_64": [],
    "//conditions:default": [
        "-pthread",
        "-lm",
        "-ldl",
    ],
})

PLATFORM_TAGS = {
    "@bazel_tools//src/conditions:windows": ["windows"],
    "@bazel_tools//src/conditions:darwin_x86_64": ["macos"],
    "//conditions:default": ["linux"],
}

# Sanitize a dependency so that it works correctly from code that includes it as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def plaidml_cc_library(copts = [], **kwargs):
    native.cc_library(copts = PLAIDML_COPTS + copts, **kwargs)

def plaidml_objc_library(copts = [], linkopts = [], **kwargs):
    native.objc_library(copts = PLAIDML_COPTS + copts + ["-Wno-shorten-64-to-32"], **kwargs)

def plaidml_cc_binary(copts = [], linkopts = [], **kwargs):
    native.cc_binary(copts = PLAIDML_COPTS + copts, linkopts = PLAIDML_LINKOPTS + linkopts, **kwargs)

def plaidml_cc_test(
        name,
        args = [],
        copts = [],
        deps = [],
        data = [],
        linkopts = [],
        toolchains = [],
        visibility = [],
        **kwargs):
    native.cc_test(
        name = name,
        args = args,
        copts = PLAIDML_COPTS + copts,
        deps = deps + [clean_dep("//pmlc/testing:gtest_main")],
        data = data,
        linkopts = PLAIDML_LINKOPTS + linkopts,
        toolchains = toolchains,
        visibility = visibility,
        **kwargs
    )
    _plaidml_args(
        name = name + "_args",
        args = args,
        data = data,
        toolchains = toolchains,
        visibility = visibility,
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
            default = clean_dep("//bzl:version.tpl.cc"),
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
            default = clean_dep("//bzl:version.tpl.py"),
            allow_single_file = True,
        ),
    },
    output_to_genfiles = True,
    outputs = {"version_file": "_version.py"},
    implementation = _plaidml_version_impl,
)

def _shlib_name_patterns(name):
    return {
        "@bazel_tools//src/conditions:windows": ["{}.dll".format(name)],
        "@bazel_tools//src/conditions:darwin_x86_64": ["lib{}.dylib".format(name)],
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

def _plaidml_settings_impl(ctx):
    return [
        platform_common.TemplateVariableInfo({
            "plaidml_device": ctx.attr._device[BuildSettingInfo].value,
            "plaidml_target": ctx.attr._target[BuildSettingInfo].value,
        }),
    ]

plaidml_settings = rule(
    attrs = {
        "_device": attr.label(default = "//plaidml:device"),
        "_target": attr.label(default = "//plaidml:target"),
    },
    implementation = _plaidml_settings_impl,
)

def _plaidml_target(settings, attr):
    return {"//plaidml:target": attr.plaidml_target}

# Defines a configuration transition to a new //plaidml:target.
#
# When used as a transition in the Bazel action graph, this causes
# rules to be built with a different //plaidml:target configuration
# (reusing build products that do not depend on //plaidml:target).
#
# See
# https://docs.bazel.build/versions/master/skylark/config.html#user-defined-transitions
# for more details on how this works.
plaidml_target = transition(
    implementation = _plaidml_target,
    inputs = [],
    outputs = ["//plaidml:target"],
)

_ArgInfo = provider(fields = ["args"])

def _plaidml_args_impl(ctx):
    args = ctx.attr.args
    args = [ctx.expand_location(arg, ctx.attr.data) for arg in args]
    args = [ctx.expand_make_variables("args", arg, {}) for arg in args]
    return [_ArgInfo(args = args)]

_plaidml_args = rule(
    attrs = {
        "args": attr.string_list(),
        "data": attr.label_list(allow_files = True),
    },
    implementation = _plaidml_args_impl,
)

def _plaidml_target_test_builder_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file._tpl,
        output = ctx.outputs.executable,
        substitutions = {},
        is_executable = True,
    )

    runfiles = ctx.runfiles()
    for test in ctx.attr.tests:
        runfiles = runfiles.merge(test[DefaultInfo].default_runfiles)

    return [DefaultInfo(runfiles = runfiles, executable = ctx.outputs.executable)]

# A rule to create a target-specific test builder script, used as a
# tool for packaging target-specific tests for use outside of the
# Bazel environment.
#
# Note that the target-specific tests are built using a transition to
# an explicit //plaidml:target configuration; the command-line and
# default //plaidml:target setting are *not* used.  This makes it
# possible to build multiple target-specific tests via a single bazel
# invocation.
_plaidml_target_test_builder = rule(
    attrs = {
        "plaidml_target": attr.string(mandatory = True),
        "tests": attr.label_list(allow_empty = False, cfg = plaidml_target),
        "_tpl": attr.label(default = "//bzl:target_test_builder.tpl.py", allow_single_file = True),

        # Allow this rule to use transitions (which bazel is fairly
        # careful about, since transitions have the potential to
        # explode the build graph).
        #
        # TODO: This has been renamed from "whitelist" to "allowlist"
        # in more recent versions of bazel.
        "_whitelist_function_transition": attr.label(
            default = "@bazel_tools//tools/whitelists/function_transition_whitelist",
        ),
    },
    outputs = {
        "executable": "%{name}.py",
    },
    implementation = _plaidml_target_test_builder_impl,
)

def _plaidml_target_test_runner_impl(ctx):
    # Build the test runner.
    script = [
        "#!/usr/bin/env python",
        "",
        "import os",
        "from pathlib import Path",
        "import subprocess",
        "",
        "p = Path(__file__).parents[0]",
        "os.chdir(p)",
        "os.environ['RUNFILES_DIR'] = str(p)",
        "os.environ['PLAIDML_TARGET'] = '{}'".format(ctx.attr.plaidml_target),
        "",
    ]

    for test, args in zip(ctx.attr.tests, [arg[_ArgInfo].args for arg in ctx.attr.args]):
        workspace = test.label.workspace_name
        if not workspace:
            workspace = ctx.workspace_name
        command = [test.files_to_run.executable.short_path] + args
        script.append("subprocess.check_call(['{}'], cwd='{}')".format("', '".join(command), workspace))

    ctx.actions.write(ctx.outputs.executable, "\n".join(script), is_executable = True)

    return [DefaultInfo(executable = ctx.outputs.executable)]

# A rule to create a target-specific test runner script, which is
# packaged into an out-of-bazel target-specific unit testing package.
#
# Note that the target-specific tests are built using a transition to
# an explicit //plaidml:target configuration; the command-line and
# default //plaidml:target setting are *not* used.  This makes it
# possible to build multiple target-specific tests via a single bazel
# invocation.
_plaidml_target_test_runner = rule(
    attrs = {
        "plaidml_target": attr.string(mandatory = True),
        "tests": attr.label_list(allow_empty = False, cfg = plaidml_target),
        "args": attr.label_list(allow_empty = False, cfg = plaidml_target, providers = [_ArgInfo]),

        # Allow this rule to use transitions (which bazel is fairly
        # careful about, since transitions have the potential to
        # explode the build graph).
        #
        # TODO: This has been renamed from "whitelist" to "allowlist"
        # in more recent versions of bazel.
        "_whitelist_function_transition": attr.label(
            default = "@bazel_tools//tools/whitelists/function_transition_whitelist",
        ),
    },
    outputs = {
        "executable": "%{name}.py",
    },
    implementation = _plaidml_target_test_runner_impl,
)

def _plaidml_target_test_package_impl(ctx):
    # Package it all up.
    ctx.actions.run(
        outputs = [ctx.outputs.archive],
        tools = [ctx.executable.builder],
        inputs = [ctx.executable.runner],
        executable = ctx.executable.builder,
        arguments = [
            ctx.outputs.archive.path,
            ctx.executable.runner.path,
        ],
        progress_message = "Building " + ctx.outputs.archive.short_path,
    )

# A rule to actually package up an out-of-bazel target-specific unit
# testing package, using the supplied builder and runner.
_plaidml_target_test_package = rule(
    attrs = {
        "builder": attr.label(
            executable = True,
            cfg = "host",
        ),
        "runner": attr.label(
            executable = True,
            cfg = "target",
        ),
    },
    outputs = {
        "archive": "%{name}.tar.gz",
    },
    implementation = _plaidml_target_test_package_impl,
)

def plaidml_target_test_package(name, plaidml_target, tests, tags = []):
    # To build an out-of-bazel test package:
    #
    # 1) We create a builder executable whose runfiles is the union
    #    all of the various test runfiles.
    #
    # 2) We create a runner executable that knows how to run the test.
    #
    # 3) We use the builder as a tool to build the output archive.
    #
    # When the builder is used as a tool, bazel does the work of
    # constructing a runfiles directory for it; the tool then packages
    # the contents of its own runfiles directory.
    _plaidml_target_test_builder(
        name = name + "_builder",
        testonly = True,
        plaidml_target = plaidml_target,
        tests = tests,
        tags = tags,
    )
    _plaidml_target_test_runner(
        name = name + "_runner",
        testonly = True,
        plaidml_target = plaidml_target,
        tests = tests,
        args = [test + "_args" for test in tests],
        tags = tags,
    )
    _plaidml_target_test_package(
        name = name,
        testonly = True,
        builder = name + "_builder",
        runner = name + "_runner",
        tags = tags,
    )
