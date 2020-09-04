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
