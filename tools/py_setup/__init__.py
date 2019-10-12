# Copyright 2019 Intel Corporation.

import os
import pathlib
import shutil
import sys
import zipfile


def bazel_stage():
    src_root = pathlib.Path(os.getenv('BZL_SRC'))
    tgt_root = pathlib.Path(os.getenv('BZL_TGT'))
    tgt_path = tgt_root / 'tmp'
    shutil.rmtree(tgt_path)
    if sys.platform == 'win32':
        stage = tgt_root / 'stage'
        with zipfile.ZipFile(src_root.with_suffix('.zip')) as zf:
            zf.extractall(stage)
        src_path = stage / 'runfiles'
        src_path.rename(tgt_path)
    else:
        src_path = src_root.with_suffix('.runfiles')
        shutil.copytree(src_path, tgt_path)
    os.chdir(tgt_path)
