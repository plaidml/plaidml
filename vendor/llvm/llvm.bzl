"""This file contains BUILD extensions for generating source code from LLVM's table definition files using the TableGen tool.

See http://llvm.org/cmds/tblgen.html for more information on the TableGen
tool.
TODO(chandlerc): Currently this expresses include-based dependencies as
"sources", and has no transitive understanding due to these files not being
correctly understood by the build system.
"""

def _dict_add(*dictionaries):
    """Returns a new `dict` that has all the entries of the given dictionaries.

    If the same key is present in more than one of the input dictionaries, the
    last of them in the argument list overrides any earlier ones.

    This function is designed to take zero or one arguments as well as multiple
    dictionaries, so that it follows arithmetic identities and callers can avoid
    special cases for their inputs: the sum of zero dictionaries is the empty
    dictionary, and the sum of a single dictionary is a copy of itself.

    Re-implemented here to avoid adding a dependency on skylib.

    Args:
      *dictionaries: Zero or more dictionaries to be added.

    Returns:
      A new `dict` that has all the entries of the given dictionaries.
    """
    result = {}
    for d in dictionaries:
        result.update(d)
    return result

def gentbl(name, tblgen, td_file, td_srcs, tbl_outs, library = True, **kwargs):
    """gentbl() generates tabular code from a table definition file.

    Args:
      name: The name of the build rule for use in dependencies.
      tblgen: The binary used to produce the output.
      td_file: The primary table definitions file.
      td_srcs: A list of table definition files included transitively.
      tbl_outs: A list of tuples (opts, out), where each opts is a string of
        options passed to tblgen, and the out is the corresponding output file
        produced.
      library: Whether to bundle the generated files into a library.
      **kwargs: Keyword arguments to pass to subsidiary cc_library() rule.
    """
    if td_file not in td_srcs:
        td_srcs += [td_file]
    includes = []
    for (opts, out) in tbl_outs:
        outdir = out[:out.rindex("/")]
        if outdir not in includes:
            includes.append(outdir)
        rule_suffix = "_".join(opts.replace("-", "_").replace("=", "_").split(" "))
        native.genrule(
            name = "%s_%s_genrule" % (name, rule_suffix),
            srcs = td_srcs,
            outs = [out],
            tools = [tblgen],
            message = "Generating code from table: %s" % td_file,
            cmd = (("$(location %s) " + "-I external/llvm-project/llvm/include " +
                    "-I external/llvm-project/clang/include " +
                    "-I $$(dirname $(location %s)) " + ("%s $(location %s) --long-string-literals=0 " +
                                                        "-o $@")) % (
                tblgen,
                td_file,
                opts,
                td_file,
            )),
        )

    # For now, all generated files can be assumed to comprise public interfaces.
    # If this is not true, you should specify library = False
    # and list the generated '.inc' files in "srcs".
    if library:
        native.cc_library(
            name = name,
            textual_hdrs = [f for (_, f) in tbl_outs],
            includes = includes,
            **kwargs
        )

def llvm_target_cmake_vars(native_arch, target_triple):
    return {
        "LLVM_HOST_TRIPLE": target_triple,
        "LLVM_DEFAULT_TARGET_TRIPLE": target_triple,
        "LLVM_NATIVE_ARCH": native_arch,
    }

def _quote(s):
    """Quotes the given string for use in a shell command.

    This function double-quotes the given string (in case it contains spaces or
    other special characters) and escapes any special characters (dollar signs,
    double-quotes, and backslashes) that may be present.

    Args:
      s: The string to quote.

    Returns:
      An escaped and quoted version of the string that can be passed to a shell
      command.
    """
    return ('"' +
            s.replace("\\", "\\\\").replace("$", "\\$").replace('"', "\\\"") +
            '"')

def cmake_var_string(cmake_vars):
    """Converts a dictionary to an input suitable for expand_cmake_vars.

    Ideally we would jist stringify in the expand_cmake_vars() rule, but select()
    interacts badly with genrules.

    TODO(phawkins): replace the genrule() with native rule and delete this rule.

    Args:
      cmake_vars: a dictionary with string keys and values that are convertable to
        strings.

    Returns:
      cmake_vars in a form suitable for passing to expand_cmake_vars.
    """
    return " ".join([
        _quote("{}={}".format(k, str(v)))
        for (k, v) in cmake_vars.items()
    ])

def expand_cmake_vars(name, src, dst, cmake_vars):
    """Expands #cmakedefine, #cmakedefine01, and CMake variables in a text file.

    Args:
      name: the name of the rule
      src: the input of the rule
      dst: the output of the rule
      cmake_vars: a string containing the CMake variables, as generated by
        cmake_var_string.
    """
    expand_cmake_vars_tool = "@com_intel_plaidml//vendor/llvm:expand_cmake_vars"  #"@org_tensorflow//third_party/llvm:expand_cmake_vars"
    native.genrule(
        name = name,
        srcs = [src],
        tools = [expand_cmake_vars_tool],
        outs = [dst],
        cmd = ("$(location {}) ".format(expand_cmake_vars_tool) + cmake_vars +
               "< $< > $@"),
    )

# TODO(phawkins): the set of CMake variables was hardcoded for expediency.
# However, we should really detect many of these via configure-time tests.

# The set of CMake variables common to all targets.
cmake_vars = {
    # LLVM features
    "ENABLE_BACKTRACES": 1,
    "LLVM_BINDIR": "/dev/null",
    "LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING": 0,
    "LLVM_ENABLE_ABI_BREAKING_CHECKS": 0,
    "LLVM_ENABLE_THREADS": 1,
    "LLVM_ENABLE_ZLIB": 1,
    "LLVM_HAS_ATOMICS": 1,
    "LLVM_INCLUDEDIR": "/dev/null",
    "LLVM_INFODIR": "/dev/null",
    "LLVM_MANDIR": "/dev/null",
    "LLVM_NATIVE_TARGET": 1,
    "LLVM_NATIVE_TARGETINFO": 1,
    "LLVM_NATIVE_TARGETMC": 1,
    "LLVM_NATIVE_ASMPRINTER": 1,
    "LLVM_NATIVE_ASMPARSER": 1,
    "LLVM_NATIVE_DISASSEMBLER": 1,
    "LLVM_PREFIX": "/dev/null",
    "LLVM_USE_INTEL_JITEVENTS": 1,
    "LLVM_VERSION_MAJOR": 0,
    "LLVM_VERSION_MINOR": 0,
    "LLVM_VERSION_PATCH": 0,
    "PACKAGE_NAME": "llvm",
    "PACKAGE_STRING": "llvm plaidml-trunk",
    "PACKAGE_VERSION": "plaidml-trunk",
    "RETSIGTYPE": "void",
}

# The set of CMake variables common to POSIX targets.
posix_cmake_vars = {
    # Headers
    "HAVE_DIRENT_H": 1,
    "HAVE_DLFCN_H": 1,
    "HAVE_ERRNO_H": 1,
    "HAVE_EXECINFO_H": 1,
    "HAVE_FCNTL_H": 1,
    "HAVE_INTTYPES_H": 1,
    "HAVE_PTHREAD_H": 1,
    "HAVE_SIGNAL_H": 1,
    "HAVE_STDINT_H": 1,
    "HAVE_SYSEXITS_H": 1,
    "HAVE_SYS_IOCTL_H": 1,
    "HAVE_SYS_MMAN_H": 1,
    "HAVE_SYS_PARAM_H": 1,
    "HAVE_SYS_RESOURCE_H": 1,
    "HAVE_SYS_STAT_H": 1,
    "HAVE_SYS_TIME_H": 1,
    "HAVE_SYS_TYPES_H": 1,
    "HAVE_TERMIOS_H": 1,
    "HAVE_UNISTD_H": 1,
    "HAVE_ZLIB_H": 1,

    # Features
    "HAVE_BACKTRACE": 1,
    "BACKTRACE_HEADER": "execinfo.h",
    "HAVE_DLOPEN": 1,
    "HAVE_FUTIMES": 1,
    "HAVE_GETCWD": 1,
    "HAVE_GETPAGESIZE": 1,
    "HAVE_GETRLIMIT": 1,
    "HAVE_GETRUSAGE": 1,
    "HAVE_GETTIMEOFDAY": 1,
    "HAVE_INT64_T": 1,
    "HAVE_ISATTY": 1,
    "HAVE_LIBEDIT": 0,  # (PlaidML)
    "HAVE_LIBPTHREAD": 1,
    "HAVE_LIBZ": 1,
    "HAVE_MKDTEMP": 1,
    "HAVE_MKSTEMP": 1,
    "HAVE_MKTEMP": 1,
    "HAVE_PREAD": 1,
    "HAVE_PTHREAD_GETSPECIFIC": 1,
    "HAVE_PTHREAD_MUTEX_LOCK": 1,
    "HAVE_PTHREAD_RWLOCK_INIT": 1,
    "HAVE_REALPATH": 1,
    "HAVE_SBRK": 1,
    "HAVE_SETENV": 1,
    "HAVE_SETRLIMIT": 1,
    "HAVE_SIGALTSTACK": 1,
    "HAVE_STRERROR": 1,
    "HAVE_STRERROR_R": 1,
    "HAVE_STRTOLL": 1,
    "HAVE_SYSCONF": 1,
    "HAVE_UINT64_T": 1,
    "HAVE__UNWIND_BACKTRACE": 1,

    # LLVM features
    "LLVM_ON_UNIX": 1,
    "LTDL_SHLIB_EXT": ".so",
}

# CMake variables specific to the Linux platform
linux_cmake_vars = {
    "HAVE_MALLOC_H": 1,
    "HAVE_LINK_H": 1,
    "HAVE_MALLINFO": 1,
    "HAVE_FUTIMENS": 1,
}

# CMake variables specific to the FreeBSD platform
freebsd_cmake_vars = {
    "HAVE_MALLOC_H": 1,
    "HAVE_LINK_H": 1,
}

# CMake variables specific to the Darwin (Mac OS X) platform.
darwin_cmake_vars = {
    "HAVE_MALLOC_MALLOC_H": 1,
    "HAVE_MALLOC_ZONE_STATISTICS": 1,
}

# CMake variables specific to the Windows platform.
win32_cmake_vars = {
    # Headers
    "HAVE_ERRNO_H": 1,
    "HAVE_EXECINFO_H": 1,
    "HAVE_FCNTL_H": 1,
    "HAVE_FENV_H": 1,
    "HAVE_INTTYPES_H": 1,
    "HAVE_MALLOC_H": 1,
    "HAVE_SIGNAL_H": 1,
    "HAVE_STDINT_H": 1,
    "HAVE_SYS_STAT_H": 1,
    "HAVE_SYS_TYPES_H": 1,
    "HAVE_ZLIB_H": 1,

    # Features
    "BACKTRACE_HEADER": "execinfo.h",
    "HAVE_GETCWD": 1,
    "HAVE_INT64_T": 1,
    "HAVE_STRERROR": 1,
    "HAVE_STRTOLL": 1,
    "HAVE_SYSCONF": 1,
    "HAVE_UINT64_T": 1,
    "HAVE__CHSIZE_S": 1,
    "HAVE___CHKSTK": 1,

    # MSVC specific
    "stricmp": "_stricmp",
    "strdup": "_strdup",

    # LLVM features
    "LTDL_SHLIB_EXT": ".dll",

    # ThreadPoolExecutor global destructor and thread handshaking do not work
    # on this platform when used as a DLL.
    # See: https://bugs.llvm.org/show_bug.cgi?id=44211
    "LLVM_ENABLE_THREADS": 0,
}

# Select a set of CMake variables based on the platform.
# TODO(phawkins): use a better method to select the right host triple, rather
# than hardcoding x86_64.
llvm_all_cmake_vars = select({
    "@bazel_tools//src/conditions:darwin_x86_64": cmake_var_string(
        _dict_add(
            cmake_vars,
            llvm_target_cmake_vars("X86", "x86_64-apple-darwin"),
            posix_cmake_vars,
            darwin_cmake_vars,
        ),
    ),
    # (PlaidML)
    # "@bazel_tools//src/conditions:linux_ppc64le": cmake_var_string(
    #     _dict_add(
    #         cmake_vars,
    #         llvm_target_cmake_vars("PowerPC", "powerpc64le-unknown-linux_gnu"),
    #         posix_cmake_vars,
    #         linux_cmake_vars,
    #     ),
    # ),
    "@bazel_tools//src/conditions:windows": cmake_var_string(
        _dict_add(
            cmake_vars,
            llvm_target_cmake_vars("X86", "x86_64-pc-win32"),
            win32_cmake_vars,
        ),
    ),
    # (PlaidML)
    # "@bazel_tools//src/conditions:freebsd": cmake_var_string(
    #     _dict_add(
    #         cmake_vars,
    #         llvm_target_cmake_vars("X86", "x86_64-unknown-freebsd"),
    #         posix_cmake_vars,
    #     ),
    # ),
    # (PlaidML)
    # "@bazel_tools//src/conditions:linux_s390x": cmake_var_string(
    #     _dict_add(
    #         cmake_vars,
    #         llvm_target_cmake_vars("SystemZ", "systemz-unknown-linux_gnu"),
    #         posix_cmake_vars,
    #         linux_cmake_vars,
    #     ),
    # ),
    "//conditions:default": cmake_var_string(
        _dict_add(
            cmake_vars,
            llvm_target_cmake_vars("X86", "x86_64-unknown-linux_gnu"),
            posix_cmake_vars,
            linux_cmake_vars,
        ),
    ),
})

llvm_linkopts = select({
    "@bazel_tools//src/conditions:windows": [],
    # (PlaidML)
    # "@bazel_tools//src/conditions:freebsd": ["-ldl", "-lm", "-lpthread", "-lexecinfo"],
    "//conditions:default": ["-ldl", "-lm", "-lpthread"],
})

llvm_defines = select({
    "@bazel_tools//src/conditions:windows": [
        "_CRT_SECURE_NO_DEPRECATE",
        "_CRT_SECURE_NO_WARNINGS",
        "_CRT_NONSTDC_NO_DEPRECATE",
        "_CRT_NONSTDC_NO_WARNINGS",
        "_SCL_SECURE_NO_DEPRECATE",
        "_SCL_SECURE_NO_WARNINGS",
        "UNICODE",
        "_UNICODE",
    ],
    "//conditions:default": [],
}) + [
    "LLVM_ENABLE_STATS",
    "__STDC_LIMIT_MACROS",
    "__STDC_CONSTANT_MACROS",
    "__STDC_FORMAT_MACROS",
    "LLVM_BUILD_GLOBAL_ISEL",
]

llvm_copts = select({
    "@com_intel_plaidml//:msvc": [
        "-Zc:inline",
        "-Zc:strictStrings",
        "-Zc:rvalueCast",
        "-Oi",
        "-wd4141",
        "-wd4146",
        "-wd4180",
        "-wd4244",
        "-wd4258",
        "-wd4267",
        "-wd4291",
        "-wd4345",
        "-wd4351",
        "-wd4355",
        "-wd4456",
        "-wd4457",
        "-wd4458",
        "-wd4459",
        "-wd4503",
        "-wd4624",
        "-wd4722",
        "-wd4800",
        "-wd4100",
        "-wd4127",
        "-wd4512",
        "-wd4505",
        "-wd4610",
        "-wd4510",
        "-wd4702",
        "-wd4245",
        "-wd4706",
        "-wd4310",
        "-wd4701",
        "-wd4703",
        "-wd4389",
        "-wd4611",
        "-wd4805",
        "-wd4204",
        "-wd4577",
        "-wd4091",
        "-wd4592",
        "-wd4319",
        "-wd4324",
        "-w14062",
        "-we4238",
    ],
    "//conditions:default": [],
})

# Platform specific sources for libSupport.

def llvm_support_platform_specific_srcs_glob():
    return select({
        "@bazel_tools//src/conditions:windows": native.glob([
            "lib/Support/Windows/*.inc",
            "lib/Support/Windows/*.h",
        ]),
        "//conditions:default": native.glob([
            "lib/Support/Unix/*.inc",
            "lib/Support/Unix/*.h",
        ]),
    })
