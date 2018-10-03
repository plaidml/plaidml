_CUDA_TOOLKIT_PATH = "CUDA_TOOLKIT_PATH"

_VAI_CUDA_REPO_VERSION = "VAI_CUDA_REPO_VERSION"

_VAI_NEED_CUDA = "VAI_NEED_CUDA"

_DEFAULT_CUDA_TOOLKIT_PATH = "/usr/local/cuda"

# Lookup paths for CUDA / cuDNN libraries, relative to the install directories.
#
# Paths will be tried out in the order listed below. The first successful path
# will be used. For example, when looking for the cudart libraries, the first
# attempt will be lib64/cudart inside the CUDA toolkit.
CUDA_LIB_PATHS = [
    "lib64/",
    "lib64/stubs/",
    "lib/x86_64-linux-gnu/",
    "lib/x64/",
    "lib/",
    "",
]

# Lookup paths for CUDA headers (cuda.h) relative to the CUDA toolkit directory.
CUDA_INCLUDE_PATHS = [
    "include/",
    "include/cuda/",
]

def get_cpu_value(ctx):
    os_name = ctx.os.name.lower()
    if os_name.startswith("mac os"):
        return "Darwin"
    if os_name.find("windows") != -1:
        return "Windows"
    result = ctx.execute(["uname", "-s"])
    return result.stdout.strip()

def _is_windows(ctx):
    """Returns true if the host operating system is windows."""
    return get_cpu_value(ctx) == "Windows"

def _lib_name(lib, cpu_value, version = "", static = False):
    """Constructs the platform-specific name of a library.

    Args:
        lib: The name of the library, such as "cudart"
        cpu_value: The name of the host operating system.
        version: The version of the library.
        static: True the library is static or False if it is a shared object.

    Returns:
        The platform-specific name of the library.
    """
    if cpu_value in ("Linux", "FreeBSD"):
        if static:
            return "lib%s.a" % lib
        else:
            if version:
                version = ".%s" % version
            return "lib%s.so%s" % (lib, version)
    if cpu_value == "Windows":
        return "%s.lib" % lib
    if cpu_value == "Darwin":
        if static:
            return "lib%s.a" % lib
        if version:
            version = ".%s" % version
        return "lib%s%s.dylib" % (lib, version)
    fail("Invalid cpu_value: %s" % cpu_value)

def _tpl(ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl.replace(":", "/")
    ctx.template(
        out,
        Label("@com_intel_plaidml//vendor/cuda:%s.tpl" % tpl),
        substitutions,
    )

def _cuda_toolkit_path(ctx):
    path = ctx.os.environ.get(_CUDA_TOOLKIT_PATH, _DEFAULT_CUDA_TOOLKIT_PATH)
    if not ctx.path(path).exists:
        fail("Cannot find CUDA toolkit path.")
    return str(ctx.path(path).realpath)

def _get_cuda_config(ctx):
    """Detects and returns information about the CUDA installation on the system.

    Args:
        ctx: The repository context.

    Returns:
        A struct containing the following fields:
        cuda_toolkit_path: The CUDA toolkit installation directory.
        compute_capabilities: A list of the system's CUDA compute capabilities.
        cpu_value: The name of the host operating system.
    """
    cpu_value = get_cpu_value(ctx)
    cuda_toolkit_path = _cuda_toolkit_path(ctx)
    return struct(
        cuda_toolkit_path = cuda_toolkit_path,
        # compute_capabilities = _compute_capabilities(ctx),
        cpu_value = cpu_value,
    )

def _find_cuda_include_path(ctx, cuda_config):
    """Returns the path to the directory containing cuda.h

    Args:
        ctx: The repository context.
        cuda_config: The CUDA config as returned by _get_cuda_config

    Returns:
        The path of the directory containing the CUDA headers.
    """
    cuda_toolkit_path = cuda_config.cuda_toolkit_path
    for relative_path in CUDA_INCLUDE_PATHS:
        if ctx.path("%s/%scuda.h" % (cuda_toolkit_path, relative_path)).exists:
            return ("%s/%s" % (cuda_toolkit_path, relative_path))[:-1]
    fail("Cannot find cuda.h under %s" % cuda_toolkit_path)

def _create_dummy_repository(ctx):
    cpu_value = get_cpu_value(ctx)
    _tpl(ctx, "build_defs.bzl", {
        "%{cuda_is_configured}": "False",
    })
    _tpl(ctx, "BUILD", {
        "%{cuda_driver_lib}": _lib_name("cuda", cpu_value),
        "%{nvrtc_lib}": _lib_name("nvrtc", cpu_value),
        "%{nvrtc_builtins_lib}": _lib_name("nvrtc-builtins", cpu_value),
        "%{cuda_include_genrules}": "",
        "%{cuda_headers}": "",
    })

    ctx.file("include/cuda.h", "")
    ctx.file("lib/%s" % _lib_name("cuda", cpu_value))

def _find_cuda_lib(lib, ctx, cpu_value, basedir, version = "", static = False):
    """Finds the given CUDA or cuDNN library on the system.

    Args:
        lib: The name of the library, such as "cudart"
        ctx: The repository context.
        cpu_value: The name of the host operating system.
        basedir: The install directory of CUDA or cuDNN.
        version: The version of the library.
        static: True if static library, False if shared object.

    Returns:
        Returns a struct with the following fields:
        file_name: The basename of the library found on the system.
        path: The full path to the library.
    """
    file_name = _lib_name(lib, cpu_value, version, static)
    for relative_path in CUDA_LIB_PATHS:
        path = ctx.path("%s/%s%s" % (basedir, relative_path, file_name))
        if path.exists:
            return struct(file_name = file_name, path = str(path.realpath))
    fail("Cannot find cuda library %s" % file_name)

def _find_libs(ctx, cuda_config):
    """Returns the CUDA and cuDNN libraries on the system.

    Args:
        ctx: The repository context.
        cuda_config: The CUDA config as returned by _get_cuda_config

    Returns:
        Map of library names to structs of filename and path.
    """
    cpu_value = cuda_config.cpu_value
    return {
        "cuda": _find_cuda_lib("cuda", ctx, cpu_value, cuda_config.cuda_toolkit_path),
        "nvrtc": _find_cuda_lib("nvrtc", ctx, cpu_value, cuda_config.cuda_toolkit_path),
        "nvrtc_builtins": _find_cuda_lib("nvrtc-builtins", ctx, cpu_value, cuda_config.cuda_toolkit_path),
    }

def _execute(ctx, cmdline, error_msg = None, error_details = None, empty_stdout_fine = False):
    """Executes an arbitrary shell command.

    Args:
        ctx: The repository context.
        cmdline: list of strings, the command to execute
        error_msg: string, a summary of the error if the command fails
        error_details: string, details about the error or steps to fix it
        empty_stdout_fine: bool, if True, an empty stdout result is fine, otherwise
        it's an error
    Return:
        the result of ctx.execute(cmdline)
    """
    result = ctx.execute(cmdline)
    if result.stderr or not (empty_stdout_fine or result.stdout):
        fail(
            "\n".join([
                error_msg.strip() if error_msg else "Repository command failed",
                result.stderr.strip(),
                error_details if error_details else "",
            ]),
        )
    return result

def _read_dir(ctx, src_dir):
    """Returns a string with all files in a directory.

    Finds all files inside a directory, traversing subfolders and following
    symlinks. The returned string contains the full path of all files
    separated by line breaks.
    """
    if _is_windows(ctx):
        src_dir = src_dir.replace("/", "\\")
        find_result = _execute(
            ctx,
            ["cmd.exe", "/c", "dir", src_dir, "/b", "/s", "/a-d"],
            empty_stdout_fine = True,
        )

        # src_files will be used in genrule.outs where the paths must
        # use forward slashes.
        result = find_result.stdout.replace("\\", "/")
    else:
        find_result = _execute(
            ctx,
            ["find", src_dir, "-follow", "-type", "f"],
            empty_stdout_fine = True,
        )
        result = find_result.stdout
    return result

def _norm_path(path):
    """Returns a path with '/' and remove the trailing slash."""
    path = path.replace("\\", "/")
    if path[-1] == "/":
        path = path[:-1]
    return path

def symlink_genrule_for_dir(ctx, src_dir, dest_dir, genrule_name, src_files = [], dest_files = []):
    """Returns a genrule to symlink(or copy if on Windows) a set of files.

    If src_dir is passed, files will be read from the given directory; otherwise
    we assume files are in src_files and dest_files
    """
    if src_dir != None:
        src_dir = _norm_path(src_dir)
        dest_dir = _norm_path(dest_dir)
        files = "\n".join(sorted(_read_dir(ctx, src_dir).splitlines()))

        # Create a list with the src_dir stripped to use for outputs.
        dest_files = files.replace(src_dir, "").splitlines()
        src_files = files.splitlines()
    command = []

    # We clear folders that might have been generated previously to avoid undesired inclusions
    if genrule_name == "cuda-include":
        command.append('if [ -d "$(@D)/cuda/include" ]; then rm -rf $(@D)/cuda/include; fi')
    elif genrule_name == "cuda-lib":
        command.append('if [ -d "$(@D)/cuda/lib" ]; then rm -rf $(@D)/cuda/lib; fi')
    outs = []
    for i in range(len(dest_files)):
        if dest_files[i] != "":
            # If we have only one file to link we do not want to use the dest_dir, as
            # $(@D) will include the full path to the file.
            dest = "$(@D)/" + dest_dir + dest_files[i] if len(dest_files) != 1 else "$(@D)/" + dest_files[i]
            command.append("mkdir -p $$(dirname {})".format(dest))
            command.append('cp -f "{}" "{}"'.format(src_files[i], dest))
            outs.append('        "{}{}",'.format(dest_dir, dest_files[i]))
    return _genrule(src_dir, genrule_name, command, outs)

def _genrule(src_dir, genrule_name, command, outs):
    """Returns a string with a genrule.

    Genrule executes the given command and produces the given outputs.
    """
    return "\n".join([
        "genrule(",
        '    name = "{}",'.format(genrule_name),
        "    outs = [",
    ] + outs + [
        "    ],",
        '    cmd = """',
    ] + command + [
        '    """,',
        ")",
    ])

def _create_cuda_repository(ctx):
    cpu_value = get_cpu_value(ctx)
    _tpl(ctx, "build_defs.bzl", {
        "%{cuda_is_configured}": "True",
    })

    cuda_config = _get_cuda_config(ctx)
    cuda_include_path = _find_cuda_include_path(ctx, cuda_config)
    cuda_libs = _find_libs(ctx, cuda_config)
    cuda_lib_src = []
    cuda_lib_dest = []
    for lib in cuda_libs.values():
        cuda_lib_src.append(lib.path)
        cuda_lib_dest.append("cuda/lib/" + lib.file_name)

    genrules = [
        symlink_genrule_for_dir(ctx, cuda_include_path, "cuda/include", "cuda-include"),
        symlink_genrule_for_dir(ctx, None, "", "cuda-lib", cuda_lib_src, cuda_lib_dest),
    ]

    _tpl(ctx, "BUILD", {
        "%{cuda_driver_lib}": cuda_libs["cuda"].file_name,
        "%{nvrtc_lib}": cuda_libs["nvrtc"].file_name,
        "%{nvrtc_builtins_lib}": cuda_libs["nvrtc_builtins"].file_name,
        "%{cuda_include_genrules}": "\n".join(genrules),
        "%{cuda_headers}": '":cuda-include",',
    })

def _configure_cuda_impl(ctx):
    enable_cuda = ctx.os.environ.get(_VAI_NEED_CUDA, "0").strip()
    if enable_cuda == "1":
        _create_cuda_repository(ctx)
    else:
        _create_dummy_repository(ctx)

configure_cuda = repository_rule(
    environ = [
        _CUDA_TOOLKIT_PATH,
        _VAI_CUDA_REPO_VERSION,
        _VAI_NEED_CUDA,
    ],
    implementation = _configure_cuda_impl,
)
