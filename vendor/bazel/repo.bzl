# Copyright 2017 The TensorFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for defining TensorFlow Bazel dependencies."""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "patch")

def _http_archive_impl(ctx):
    override = ctx.attr.override
    if override != None and override in ctx.os.environ:
        repo_path = ctx.os.environ[override]
        print("using override: {}={}".format(override, repo_path))
        ctx.symlink(repo_path, ctx.path("."))
        for src, tgt in ctx.attr.link_files.items():
            ctx.delete(ctx.path(tgt))
    else:
        ctx.download_and_extract(
            ctx.attr.url,
            "",
            ctx.attr.sha256,
            ctx.attr.type,
            ctx.attr.strip_prefix,
        )
        if ctx.attr.delete:
            for path in ctx.attr.delete:
                ctx.delete(ctx.path(path))
        patch(ctx)
    if ctx.attr.build_file != None:
        # Use "BUILD.bazel" to avoid conflict with third party projects that contain a
        # file or directory called "BUILD"
        buildfile_path = ctx.path("BUILD.bazel")
        ctx.symlink(ctx.attr.build_file, buildfile_path)
    for src, tgt in ctx.attr.link_files.items():
        ctx.delete(ctx.path(tgt))
        ctx.symlink(Label(src), ctx.path(tgt))

# Downloads and creates Bazel repos for dependencies.
#
# For link_files, specify each dict entry as:
# "//path/to/source:file": "localfile"
http_archive = repository_rule(
    implementation = _http_archive_impl,
    attrs = {
        "sha256": attr.string(),
        "url": attr.string(mandatory = True),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "delete": attr.string_list(),
        "build_file": attr.label(),
        "link_files": attr.string_dict(),
        "override": attr.string(),
        "patches": attr.label_list(default = []),
        "patch_tool": attr.string(default = ""),
        "patch_args": attr.string_list(default = ["-p1"]),
        "patch_cmds": attr.string_list(default = []),
        "patch_cmds_win": attr.string_list(default = []),
    },
)
