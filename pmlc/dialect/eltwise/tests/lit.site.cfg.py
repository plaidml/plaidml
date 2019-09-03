import os

config.runfiles_dir = os.getenv('RUNFILES_DIR')
config.llvm_tools_dir = os.path.join(config.runfiles_dir, 'llvm')
config.pmlc_tools_dir = os.path.join(
    config.runfiles_dir,
    'com_intel_plaidml',
    'pmlc',
    'tools',
)

import lit.llvm
lit.llvm.initialize(lit_config, config)

# Let the main config do the real work.
lit_cfg_path = os.path.join(os.path.dirname(__file__), "lit.cfg.py")
lit_config.load_config(config, lit_cfg_path)
