# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lit runner configuration."""

import os
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'MLIR ' + os.path.basename(config.mlir_test_dir)

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# test_source_root: The root path where tests are located.
config.test_source_root = config.mlir_test_dir

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.environ['RUNFILES_DIR']

llvm_config.use_default_substitutions()

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = config.mlir_tf_tools_dirs + [config.mlir_tools_dir, config.llvm_tools_dir]
tool_names = [
    'mlir-opt',
    'mlir-translate',
    'pmlc-opt',
    'pmlc-translate',
]
tools = [ToolSubst(s, unresolved='ignore') for s in tool_names]
llvm_config.add_tool_substitutions(tools, tool_dirs)
