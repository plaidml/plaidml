#!/usr/bin/env python3

import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


# Return True if running on Windows
def IsWindows():
    return os.name == 'nt'

if IsWindows():
    SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE = 0x02

    import ctypes
    CreateSymbolicLink = ctypes.windll.kernel32.CreateSymbolicLinkW
    CreateSymbolicLink.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    CreateSymbolicLink.restype = ctypes.c_ubyte

    def symlink(src, dst):
        flags = SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE
        if os.path.isdir(src):
            flags |= 1
        res = CreateSymbolicLink(dst, src, flags)
        if res == 0:
            raise ctypes.WinError()

    os.symlink = symlink

def GetWindowsPathWithUNCPrefix(path):
    """
    Adding UNC prefix after getting a normalized absolute Windows path,
    it's no-op for non-Windows platforms or if running under python2.
    """
    path = path.strip()

    # No need to add prefix for non-Windows platforms.
    # And \\?\ doesn't work in python 2
    if not IsWindows() or sys.version_info[0] < 3:
        return path

    # Lets start the unicode fun
    unicode_prefix = "\\\\?\\"
    if path.startswith(unicode_prefix):
        return path

    # os.path.abspath returns a normalized absolute path
    return unicode_prefix + os.path.abspath(path)


def CreatePythonPathEntries(python_imports, module_space):
    parts = python_imports.split(':')
    return [module_space] + ["%s/%s" % (module_space, path) for path in parts]


# Find the runfiles tree
def FindModuleSpace():
    stub_filename = os.path.abspath(sys.argv[0])
    while True:
        module_space = stub_filename + '.runfiles'
        if os.path.isdir(module_space):
            return module_space

        if IsWindows():
            runfiles_pattern = "(.*\.runfiles)\\.*"
        else:
            runfiles_pattern = "(.*\.runfiles)/.*"
        matchobj = re.match(runfiles_pattern, stub_filename)
        if matchobj:
            return matchobj.group(1)

        if not os.path.islink(stub_filename):
            break
        target = os.readlink(stub_filename)
        if os.path.isabs(target):
            stub_filename = target
        else:
            stub_filename = os.path.join(os.path.dirname(stub_filename), target)

    raise AssertionError('Cannot find .runfiles directory for %s' % sys.argv[0])


# Returns repository roots to add to the import path.
def GetRepositoriesImports(module_space, import_all):
    if import_all:
        repo_dirs = [os.path.join(module_space, d) for d in os.listdir(module_space)]
        return [d for d in repo_dirs if os.path.isdir(d)]
    return [os.path.join(module_space, "%workspace_name%")]


# Finds the runfiles manifest or the runfiles directory.
def RunfilesEnvvar(module_space):
    runfiles = os.getenv('RUNFILES_DIR')
    if runfiles:
        return ('RUNFILES_DIR', runfiles)

    # Look for the runfiles "input" manifest, argv[0] + ".runfiles/MANIFEST"
    runfiles = os.path.dirname(os.path.join(module_space, 'MANIFEST'))
    if os.path.exists(runfiles):
        return ('RUNFILES_DIR', runfiles)

    # If running in a sandbox and no environment variables are set, then
    # Look for the runfiles  next to the binary.
    if module_space.endswith('.runfiles') and os.path.isdir(module_space):
        return ('RUNFILES_DIR', module_space)

    return (None, None)


def Main():
    args = sys.argv[1:]

    new_env = {}

    module_space = FindModuleSpace()

    python_imports = '%imports%'
    python_path_entries = CreatePythonPathEntries(python_imports, module_space)
    python_path_entries += GetRepositoriesImports(module_space, %import_all%)

    python_path_entries = [GetWindowsPathWithUNCPrefix(d) for d in python_path_entries]

    old_python_path = os.environ.get('PYTHONPATH')
    python_path = os.pathsep.join(python_path_entries)
    if old_python_path:
        python_path += os.pathsep + old_python_path

    if IsWindows():
        python_path = python_path.replace("/", os.sep)

    new_env['PYTHONPATH'] = python_path
    runfiles_envkey, runfiles_envvalue = RunfilesEnvvar(module_space)
    if runfiles_envkey:
        new_env[runfiles_envkey] = runfiles_envvalue

    # Now look for my main python source file.
    # The magic string percent-main-percent is replaced with the filename of the
    # main file of the Python binary in BazelPythonSemantics.java.
    rel_path = os.path.normpath('%main%')

    manifest = {}
    manifest_filename = os.path.join(module_space, 'MANIFEST')
    with open(manifest_filename) as manifest_file:
        for line in manifest_file:
            (logical, physical) = line.split(' ', 2)
            local = os.path.join(module_space, os.path.normpath(logical))
            if not os.path.exists(local):
                if os.path.islink(local):
                    os.unlink(local)
                dir_name = os.path.dirname(local)
                if not os.path.isdir(dir_name):
                    os.makedirs(GetWindowsPathWithUNCPrefix(dir_name))
                src = GetWindowsPathWithUNCPrefix(os.path.normpath(physical.rstrip()))
                os.symlink(src, GetWindowsPathWithUNCPrefix(local))

    main_filename = os.path.join(module_space, rel_path)
    main_filename = GetWindowsPathWithUNCPrefix(main_filename)
    assert os.path.exists(main_filename), 'Cannot exec() %r: file not found.' % main_filename
    assert os.access(main_filename, os.R_OK), 'Cannot exec() %r: file not readable.' % main_filename

    cenv_dir = os.path.join(module_space, '.cenv')
    path = os.environ['PATH'].split(os.pathsep)
    if IsWindows():
        program = os.path.join(cenv_dir, 'python.exe')
        path.insert(0, os.path.join(cenv_dir, 'Scripts'))
        path.insert(0, cenv_dir)
        path.insert(0, os.path.join(cenv_dir, 'Library', 'bin'))
    else:
        bin_dir = os.path.join(cenv_dir, 'bin')
        program = os.path.join(bin_dir, 'python')
        path.insert(0, bin_dir)
    new_env['PATH'] = os.pathsep.join(path)
    new_env['CONDA_DEFAULT_ENV'] = cenv_dir
    args = [program, main_filename] + args

    os.environ.update(new_env)

    try:
        sys.stdout.flush()
        if IsWindows():
            # On Windows, os.execv doesn't handle arguments with spaces correctly,
            # and it actually starts a subprocess just like subprocess.call.
            sys.exit(subprocess.call(args))
        else:
            os.execv(args[0], args)
    except EnvironmentError:
        # This works from Python 2.4 all the way to 3.x.
        e = sys.exc_info()[1]
        # This exception occurs when os.execv() fails for some reason.
        if not getattr(e, 'filename', None):
            e.filename = program  # Add info to error message
        raise


if __name__ == '__main__':
    Main()
