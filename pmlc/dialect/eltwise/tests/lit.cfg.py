# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'pmlc-scalar'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.td', '.mlir']

llvm_config.use_default_substitutions()

tool_dirs = [
    os.path.join(config.pmlc_tools_dir, 'pmlc-opt'),
]

tools = [
    'pmlc-opt',
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
