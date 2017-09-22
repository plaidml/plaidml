# Bazel build file for gflags
#
# See INSTALL.md for instructions for adding gflags to a Bazel workspace.

licenses(["notice"])

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

config_setting(
  name = "x64linux",
  values = {"cpu": "k8"},
  visibility = ["//visibility:public"],
)

config_setting(
  name = "x64_windows",
  values = {"cpu": "x64_windows"},
  visibility = ["//visibility:public"],
)

COPTS_ALL = [
    # The config.h gets generated to the package directory of
    # GENDIR, and we don't want to put it into the includes
    # otherwise the dependent may pull it in by accident.
    "-I$(GENDIR)/" + PACKAGE_NAME,
    "-Wno-sign-compare",
    "-DHAVE_STDINT_H",
    "-DHAVE_SYS_TYPES_H",
    "-DHAVE_INTTYPES_H",
    "-DHAVE_SYS_STAT_H",
    "-DHAVE_UNISTD_H",
    "-DHAVE_STRTOLL",
    "-DHAVE_STRTOQ",
    "-DHAVE_PTHREAD",
    "-DHAVE_RWLOCK",
    "-DGFLAGS_INTTYPES_FORMAT_C99",
]

COPTS_LINUX = COPTS_ALL + ["-DHAVE_FNMATCH_H",]

COPTS_WINDOWS = [
  "-DOS_WINDOWS",
  "-DHAVE_STDINT_H",
  "-DHAVE_INTTYPES_H",
  "-DGFLAGS_INTTYPES_FORMAT_C99",
]

cc_library(
  name = "gflags",
  srcs = [
    "src/gflags.cc",
    "src/gflags_completions.cc",
    "src/gflags_reporting.cc",
    "src/mutex.h",
    "src/util.h",
    ":config_h",
    ":gflags_completions_h",
    ":gflags_declare_h",
    ":gflags_h",
  ],
  hdrs = [":includes"],
  copts = select({
    "//:x64linux": COPTS_LINUX,
    "//:x64_windows": COPTS_WINDOWS,
    "//conditions:default": COPTS_ALL,
  }),
  linkopts = select({
    "//:darwin": [],
    "//:x64_windows": [],
    "//conditions:default": ["-pthread"],
  }),
  includes = [
    "",
    "include",
  ],
  visibility = ["//visibility:public"],
)

genrule(
  name = "config_h",
  srcs = [
    "src/config.h.in",
  ],
  outs = [
    "config.h",
  ],
  cmd = "awk '{ gsub(/^#cmakedefine/, \"//cmakedefine\"); print; }' $(<) > $(@)",
)

genrule(
  name = "gflags_h",
  srcs = [
    "src/gflags.h.in",
  ],
  outs = [
    "gflags.h",
  ],
  cmd = "awk '{ gsub(/@(GFLAGS_ATTRIBUTE_UNUSED|INCLUDE_GFLAGS_NS_H)@/, \"\"); print; }' $(<) > $(@)",
)

genrule(
  name = "gflags_completions_h",
  srcs = [
    "src/gflags_completions.h.in",
  ],
  outs = [
    "gflags_completions.h",
  ],
  cmd = "awk '{ gsub(/@GFLAGS_NAMESPACE@/, \"gflags\"); print; }' $(<) > $(@)",
)

genrule(
  name = "gflags_declare_h",
  srcs = [
    "src/gflags_declare.h.in",
  ],
  outs = [
    "gflags_declare.h",
  ],
  cmd = ("awk '{ " +
         "gsub(/@GFLAGS_NAMESPACE@/, \"gflags\"); " +
         "gsub(/@(HAVE_STDINT_H|HAVE_SYS_TYPES_H|HAVE_INTTYPES_H|GFLAGS_INTTYPES_FORMAT_C99)@/, \"1\"); " +
         "gsub(/@([A-Z0-9_]+)@/, \"0\"); " +
         "print; }' $(<) > $(@)"),
)

genrule(
  name = "includes",
  srcs = [
    ":gflags_h",
    ":gflags_declare_h",
  ],
  outs = [
    "include/gflags/gflags.h",
    "include/gflags/gflags_declare.h",
  ],
  cmd = "mkdir -p $(@D)/include/gflags && cp $(SRCS) $(@D)/include/gflags",
)
