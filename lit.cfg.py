# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'PML'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.pml_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    'Inputs',
    'CMakeLists.txt',
    'README.txt',
    'LICENSE.txt',
    'lit.cfg.py',
    'lit.site.cfg.py',
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.pml_obj_root, 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [
    config.pml_bin_dir,
    config.llvm_tools_dir,
]
tools = [
    'pmlc-opt',
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment['FILECHECK_OPTS'] = "-enable-var-scope"

# LLVM can be configured with an empty default triple
# by passing ` -DLLVM_DEFAULT_TARGET_TRIPLE="" `.
# This is how LLVM filters tests that require the host target
# to be available for JIT tests.
if config.target_triple:
    config.available_features.add('default_triple')

# Add the python path for both the source and binary tree.
# Note that presently, the python sources come from the source tree and the
# binaries come from the build tree. This should be unified to the build tree
# by copying/linking sources to build.
# if config.enable_bindings_python:
#     llvm_config.with_environment(
#         'PYTHONPATH',
#         [
#             # TODO: Don't reference the llvm_obj_root here: the invariant is that
#             # the python/ must be at the same level of the lib directory
#             # where libMLIR.so is installed. This is presently not optimal from a
#             # project separation perspective and a discussion on how to better
#             # segment MLIR libraries needs to happen. See also
#             # lib/Bindings/Python/CMakeLists.txt for where this is set up.
#             os.path.join(config.llvm_obj_root, 'python'),
#         ],
#         append_path=True)
