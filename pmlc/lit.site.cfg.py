# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Lit runner site configuration."""

import os
import lit.llvm

config.llvm_tools_dir = os.path.join(os.environ['TEST_SRCDIR'], 'llvm-project', 'llvm')
config.mlir_obj_root = os.path.join(os.environ['TEST_SRCDIR'])
config.mlir_tools_dir = os.path.join(os.environ['TEST_SRCDIR'], 'llvm-project', 'mlir')
config.suffixes = ['.td', '.mlir', '.pbtxt', '.cc', '.py']

mlir_pmlc_tools_dirs = [
    'pmlc/tools/pmlc-jit',
    'pmlc/tools/pmlc-opt',
    'pmlc/tools/pmlc-tblgen',
    'pmlc/tools/pmlc-translate',
    'pmlc/tools/pmlc-vulkan-runner',
    'plaidml/edsl/tests',
]
config.mlir_pmlc_tools_dirs = [
    os.path.join(os.environ['TEST_SRCDIR'], os.environ['TEST_WORKSPACE'], s)
    for s in mlir_pmlc_tools_dirs
]
test_dir = os.environ['TEST_TARGET']
test_dir = test_dir.strip('/').rsplit(':', 1)[0]
config.mlir_test_dir = os.path.join(
    os.environ['TEST_SRCDIR'],
    os.environ['TEST_WORKSPACE'],
    test_dir,
)
config.lit_tools_dir = ''  # This is needed for Windows support
lit.llvm.initialize(lit_config, config)

# Let the main config do the real work.
lit_config.load_config(
    config,
    os.path.join(
        os.path.join(
            os.environ['TEST_SRCDIR'],
            os.environ['TEST_WORKSPACE'],
            'pmlc/lit.cfg.py',
        )))
