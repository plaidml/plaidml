# Copyright 2016 The Bazel Authors. All rights reserved.
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
"""Rules for downloading files and archives over HTTP.

### Setup

To use these rules, load them in your `WORKSPACE` file as follows:

```python
load(
    "@bazel_tools//tools/build_defs/repo:http.bzl",
    "http_archive",
    "http_file",
    "http_jar",
)
```

These rules are improved versions of the native http rules and will eventually
replace the native rules.
"""

load(
    ":utils.bzl",
    "patch",
    "read_netrc",
    "update_attrs",
    "use_netrc",
    "workspace_and_buildfile",
)

def _get_auth(ctx, urls):
    """Given the list of URLs obtain the correct auth dict."""
    if ctx.attr.netrc:
        netrc = read_netrc(ctx, ctx.attr.netrc)
        return use_netrc(netrc, urls)

    if "HOME" in ctx.os.environ:
        if not ctx.os.name.startswith("windows"):
            netrcfile = "%s/.netrc" % (ctx.os.environ["HOME"],)
            if ctx.execute(["test", "-f", netrcfile]).return_code == 0:
                netrc = read_netrc(ctx, netrcfile)
                return use_netrc(netrc, urls)

        # TODO: Search at a similarly canonical place for Windows as well

    return {}

def _http_archive_impl(ctx):
    """Implementation of the http_archive rule."""
    if not ctx.attr.url and not ctx.attr.urls:
        fail("At least one of url and urls must be provided")
    if ctx.attr.build_file and ctx.attr.build_file_content:
        fail("Only one of build_file and build_file_content can be provided.")

    env_key = "BAZEL_" + ctx.name.upper() + "_PATH"
    if env_key in ctx.os.environ:
        repo_path = ctx.os.environ[env_key]
        script_path = ctx.path(Label("@com_intel_plaidml//vendor/bazel:redirect.py"))

        result = ctx.execute([
            ctx.which("python"),
            script_path,
            repo_path,
            ctx.path("."),
        ], quiet = False)

        if not result.return_code == 0:
            fail("dev_http_archive failure: %s\n" % result.stderr)

        workspace_and_buildfile(ctx)

        return update_attrs(ctx.attr, _http_archive_attrs.keys(), {})

    all_urls = []
    if ctx.attr.urls:
        all_urls = ctx.attr.urls
    if ctx.attr.url:
        all_urls = [ctx.attr.url] + all_urls

    auth = _get_auth(ctx, all_urls)

    download_info = ctx.download_and_extract(
        all_urls,
        "",
        ctx.attr.sha256,
        ctx.attr.type,
        ctx.attr.strip_prefix,
        canonical_id = ctx.attr.canonical_id,
        auth = auth,
    )
    patch(ctx)
    workspace_and_buildfile(ctx)

    return update_attrs(ctx.attr, _http_archive_attrs.keys(), {"sha256": download_info.sha256})

_HTTP_FILE_BUILD = """
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "file",
    srcs = ["{}"],
)
"""

def _http_file_impl(ctx):
    """Implementation of the http_file rule."""
    repo_root = ctx.path(".")
    forbidden_files = [
        repo_root,
        ctx.path("WORKSPACE"),
        ctx.path("BUILD"),
        ctx.path("BUILD.bazel"),
        ctx.path("file/BUILD"),
        ctx.path("file/BUILD.bazel"),
    ]
    downloaded_file_path = ctx.attr.downloaded_file_path
    download_path = ctx.path("file/" + downloaded_file_path)
    if download_path in forbidden_files or not str(download_path).startswith(str(repo_root)):
        fail("'%s' cannot be used as downloaded_file_path in http_file" % ctx.attr.downloaded_file_path)
    auth = _get_auth(ctx, ctx.attr.urls)
    download_info = ctx.download(
        ctx.attr.urls,
        "file/" + downloaded_file_path,
        ctx.attr.sha256,
        ctx.attr.executable,
        auth = auth,
    )
    ctx.file("WORKSPACE", "workspace(name = \"{name}\")".format(name = ctx.name))
    ctx.file("file/BUILD", _HTTP_FILE_BUILD.format(downloaded_file_path))

    return update_attrs(ctx.attr, _http_file_attrs.keys(), {"sha256": download_info.sha256})

_HTTP_JAR_BUILD = """
package(default_visibility = ["//visibility:public"])

java_import(
  name = 'jar',
  jars = ['downloaded.jar'],
  visibility = ['//visibility:public'],
)

filegroup(
  name = 'file',
  srcs = ['downloaded.jar'],
  visibility = ['//visibility:public'],
)

"""

def _http_jar_impl(ctx):
    """Implementation of the http_jar rule."""
    all_urls = []
    if ctx.attr.urls:
        all_urls = ctx.attr.urls
    if ctx.attr.url:
        all_urls = [ctx.attr.url] + all_urls
    auth = _get_auth(ctx, all_urls)
    download_info = ctx.download(
        all_urls,
        "jar/downloaded.jar",
        ctx.attr.sha256,
        auth = auth,
    )
    ctx.file("WORKSPACE", "workspace(name = \"{name}\")".format(name = ctx.name))
    ctx.file("jar/BUILD", _HTTP_JAR_BUILD)
    return update_attrs(ctx.attr, _http_jar_attrs.keys(), {"sha256": download_info.sha256})

_http_archive_attrs = {
    "url": attr.string(
        doc =
            """A URL to a file that will be made available to Bazel.

This must be a file, http or https URL. Redirections are followed.
Authentication is not supported.

This parameter is to simplify the transition from the native http_archive
rule. More flexibility can be achieved by the urls parameter that allows
to specify alternative URLs to fetch from.
""",
    ),
    "urls": attr.string_list(
        doc =
            """A list of URLs to a file that will be made available to Bazel.

Each entry must be a file, http or https URL. Redirections are followed.
Authentication is not supported.""",
    ),
    "sha256": attr.string(
        doc = """The expected SHA-256 of the file downloaded.

This must match the SHA-256 of the file downloaded. _It is a security risk
to omit the SHA-256 as remote files can change._ At best omitting this
field will make your build non-hermetic. It is optional to make development
easier but should be set before shipping.""",
    ),
    "netrc": attr.string(
        doc = "Location of the .netrc file to use for authentication",
    ),
    "canonical_id": attr.string(
        doc = """A canonical id of the archive downloaded

If specified and non-empty, bazel will not take a the archive from cache,
unless is was added to the cache by a request with the same canonical id.
""",
    ),
    "strip_prefix": attr.string(
        doc = """A directory prefix to strip from the extracted files.

Many archives contain a top-level directory that contains all of the useful
files in archive. Instead of needing to specify this prefix over and over
in the `build_file`, this field can be used to strip it from all of the
extracted files.

For example, suppose you are using `foo-lib-latest.zip`, which contains the
directory `foo-lib-1.2.3/` under which there is a `WORKSPACE` file and are
`src/`, `lib/`, and `test/` directories that contain the actual code you
wish to build. Specify `strip_prefix = "foo-lib-1.2.3"` to use the
`foo-lib-1.2.3` directory as your top-level directory.

Note that if there are files outside of this directory, they will be
discarded and inaccessible (e.g., a top-level license file). This includes
files/directories that start with the prefix but are not in the directory
(e.g., `foo-lib-1.2.3.release-notes`). If the specified prefix does not
match a directory in the archive, Bazel will return an error.""",
    ),
    "type": attr.string(
        doc = """The archive type of the downloaded file.

By default, the archive type is determined from the file extension of the
URL. If the file has no extension, you can explicitly specify one of the
following: `"zip"`, `"jar"`, `"war"`, `"tar"`, `"tar.gz"`, `"tgz"`,
`"tar.xz"`, or `tar.bz2`.""",
    ),
    "patches": attr.label_list(
        default = [],
        doc =
            "A list of files that are to be applied as patches after " +
            "extracting the archive. By default, it uses the Bazel-native patch implementation " +
            "which doesn't support fuzz match and binary patch, but Bazel will fall back to use " +
            "patch command line tool if `patch_tool` attribute is specified or there are " +
            "arguments other than `-p` in `patch_args` attribute.",
    ),
    "patch_tool": attr.string(
        default = "",
        doc = "The patch(1) utility to use. If this is specified, Bazel will use the specifed " +
              "patch tool instead of the Bazel-native patch implementation.",
    ),
    "patch_args": attr.string_list(
        default = ["-p0"],
        doc =
            "The arguments given to the patch tool. Defaults to -p0, " +
            "however -p1 will usually be needed for patches generated by " +
            "git. If multiple -p arguments are specified, the last one will take effect." +
            "If arguments other than -p are specified, Bazel will fall back to use patch " +
            "command line tool instead of the Bazel-native patch implementation. When falling " +
            "back to patch command line tool and patch_tool attribute is not specified, " +
            "`patch` will be used.",
    ),
    "patch_cmds": attr.string_list(
        default = [],
        doc = "Sequence of Bash commands to be applied on Linux/Macos after patches are applied.",
    ),
    "patch_cmds_win": attr.string_list(
        default = [],
        doc = "Sequence of Powershell commands to be applied on Windows after patches are " +
              "applied. If this attribute is not set, patch_cmds will be executed on Windows, " +
              "which requires Bash binary to exist.",
    ),
    "build_file": attr.label(
        allow_single_file = True,
        doc =
            "The file to use as the BUILD file for this repository." +
            "This attribute is an absolute label (use '@//' for the main " +
            "repo). The file does not need to be named BUILD, but can " +
            "be (something like BUILD.new-repo-name may work well for " +
            "distinguishing it from the repository's actual BUILD files. " +
            "Either build_file or build_file_content can be specified, but " +
            "not both.",
    ),
    "build_file_content": attr.string(
        doc =
            "The content for the BUILD file for this repository. " +
            "Either build_file or build_file_content can be specified, but " +
            "not both.",
    ),
    "workspace_file": attr.label(
        doc =
            "The file to use as the `WORKSPACE` file for this repository. " +
            "Either `workspace_file` or `workspace_file_content` can be " +
            "specified, or neither, but not both.",
    ),
    "workspace_file_content": attr.string(
        doc =
            "The content for the WORKSPACE file for this repository. " +
            "Either `workspace_file` or `workspace_file_content` can be " +
            "specified, or neither, but not both.",
    ),
}

dev_http_archive = repository_rule(
    implementation = _http_archive_impl,
    attrs = _http_archive_attrs,
    doc =
        """Downloads a Bazel repository as a compressed archive file, decompresses it,
and makes its targets available for binding.

It supports the following file extensions: `"zip"`, `"jar"`, `"war"`, `"tar"`,
`"tar.gz"`, `"tgz"`, `"tar.xz"`, and `tar.bz2`.

Examples:
  Suppose the current repository contains the source code for a chat program,
  rooted at the directory `~/chat-app`. It needs to depend on an SSL library
  which is available from http://example.com/openssl.zip. This `.zip` file
  contains the following directory structure:

  ```
  WORKSPACE
  src/
    openssl.cc
    openssl.h
  ```

  In the local repository, the user creates a `openssl.BUILD` file which
  contains the following target definition:

  ```python
  cc_library(
      name = "openssl-lib",
      srcs = ["src/openssl.cc"],
      hdrs = ["src/openssl.h"],
  )
  ```

  Targets in the `~/chat-app` repository can depend on this target if the
  following lines are added to `~/chat-app/WORKSPACE`:

  ```python
  load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

  http_archive(
      name = "my_ssl",
      urls = ["http://example.com/openssl.zip"],
      sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      build_file = "@//:openssl.BUILD",
  )
  ```

  Then targets would specify `@my_ssl//:openssl-lib` as a dependency.
""",
)

_http_file_attrs = {
    "executable": attr.bool(
        doc = "If the downloaded file should be made executable.",
    ),
    "downloaded_file_path": attr.string(
        default = "downloaded",
        doc = "Path assigned to the file downloaded",
    ),
    "sha256": attr.string(
        doc = """The expected SHA-256 of the file downloaded.

This must match the SHA-256 of the file downloaded. _It is a security risk
to omit the SHA-256 as remote files can change._ At best omitting this
field will make your build non-hermetic. It is optional to make development
easier but should be set before shipping.""",
    ),
    "urls": attr.string_list(
        mandatory = True,
        doc = """A list of URLs to a file that will be made available to Bazel.

Each entry must be a file, http or https URL. Redirections are followed.
Authentication is not supported.""",
    ),
    "netrc": attr.string(
        doc = "Location of the .netrc file to use for authentication",
    ),
}

http_file = repository_rule(
    implementation = _http_file_impl,
    attrs = _http_file_attrs,
    doc =
        """Downloads a file from a URL and makes it available to be used as a file
group.

Examples:
  Suppose you need to have a debian package for your custom rules. This package
  is available from http://example.com/package.deb. Then you can add to your
  WORKSPACE file:

  ```python
  load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

  http_file(
      name = "my_deb",
      urls = ["http://example.com/package.deb"],
      sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  )
  ```

  Targets would specify `@my_deb//file` as a dependency to depend on this file.
""",
)

_http_jar_attrs = {
    "sha256": attr.string(
        doc = "The expected SHA-256 of the file downloaded.",
    ),
    "url": attr.string(
        doc =
            "The URL to fetch the jar from. It must end in `.jar`.",
    ),
    "urls": attr.string_list(
        doc =
            "A list of URLS the jar can be fetched from. They have to end " +
            "in `.jar`.",
    ),
    "netrc": attr.string(
        doc = "Location of the .netrc file to use for authentication",
    ),
}

http_jar = repository_rule(
    implementation = _http_jar_impl,
    attrs = _http_jar_attrs,
    doc =
        """Downloads a jar from a URL and makes it available as java_import

Downloaded files must have a .jar extension.

Examples:
  Suppose the current repository contains the source code for a chat program, rooted at the
  directory `~/chat-app`. It needs to depend on an SSL library which is available from
  `http://example.com/openssl-0.2.jar`.

  Targets in the `~/chat-app` repository can depend on this target if the following lines are
  added to `~/chat-app/WORKSPACE`:

  ```python
  load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_jar")

  http_jar(
      name = "my_ssl",
      url = "http://example.com/openssl-0.2.jar",
      sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  )
  ```

  Targets would specify <code>@my_ssl//jar</code> as a dependency to depend on this jar.

  You may also reference files on the current system (localhost) by using "file:///path/to/file"
  if you are on Unix-based systems. If you're on Windows, use "file:///c:/path/to/file". In both
  examples, note the three slashes (`/`) -- the first two slashes belong to `file://` and the third
  one belongs to the absolute path to the file.
""",
)
