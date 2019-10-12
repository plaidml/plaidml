# Copyright 2019 Intel Corporation.

import os
import shutil
import sys


def bazel_stage():
    src = os.getenv('BZL_SRC')
    tgt = os.path.join(os.getenv('BZL_TGT'), 'tmp')
    shutil.rmtree(tgt)
    if sys.platform == 'win32':
        src = src + '.zip'
        # unzip
    else:
        src = os.path.join(src + '.runfiles', os.getenv('BZL_WORKSPACE'))
        print('src: {}'.format(src))
        print('tgt: {}'.format(tgt))
        shutil.copytree(src, tgt)
    os.chdir(tgt)
