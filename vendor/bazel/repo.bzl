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

def _is_windows(ctx):
    return ctx.os.name.lower().find("windows") != -1

def _wrap_bash_cmd(ctx, cmd):
    if _is_windows(ctx):
        bazel_sh = _get_env_var(ctx, "BAZEL_SH")
        if not bazel_sh:
            fail("BAZEL_SH environment variable is not set")
        cmd = [bazel_sh, "-l", "-c", " ".join(["\"%s\"" % s for s in cmd])]
    return cmd

def _get_env_var(ctx, name):
    if name in ctx.os.environ:
        return ctx.os.environ[name]
    else:
        return None

# Executes specified command with arguments and calls 'fail' if it exited with
# non-zero code
def _execute_and_check_ret_code(repo_ctx, cmd_and_args):
    result = repo_ctx.execute(cmd_and_args, timeout = 60)
    if result.return_code != 0:
        fail(("Non-zero return code({1}) when executing '{0}':\n" + "Stdout: {2}\n" +
              "Stderr: {3}").format(
            " ".join([str(x) for x in cmd_and_args]),
            result.return_code,
            result.stdout,
            result.stderr,
        ))

# Apply a patch_file to the repository root directory
# Runs 'patch -p1' on both Windows and Unix.
def _apply_patch(ctx, patch_file):
    patch_command = ["patch", "-p1", "-d", ctx.path("."), "-i", ctx.path(patch_file)]
    cmd = _wrap_bash_cmd(ctx, patch_command)
    _execute_and_check_ret_code(ctx, cmd)

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
        if ctx.attr.patch_file != None:
            _apply_patch(ctx, ctx.attr.patch_file)
    if ctx.attr.build_file != None:
        # Use "BUILD.bazel" to avoid conflict with third party projects that contain a
        # file or directory called "BUILD"
        buildfile_path = ctx.path("BUILD.bazel")
        ctx.symlink(ctx.attr.build_file, buildfile_path)
    for src, tgt in ctx.attr.link_files.items():
        ctx.symlink(Label(src), ctx.path(tgt))

# Downloads and creates Bazel repos for dependencies.
#
# For link_files, specify each dict entry as:
# "//path/to/source:file": "localfile"
http_archive = repository_rule(
    implementation = _http_archive_impl,
    attrs = {
        "sha256": attr.string(mandatory = True),
        "url": attr.string(mandatory = True),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "delete": attr.string_list(),
        "build_file": attr.label(),
        "patch_file": attr.label(),
        "link_files": attr.string_dict(),
        "override": attr.string(),
    },
)
