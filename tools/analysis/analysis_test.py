# Copyright 2018 Intel Corporation.

import os
import subprocess
import unittest


class AnalysisTest(unittest.TestCase):

    def testRunNotebook(self):
        cwd = os.path.abspath('../com_intel_plaidml/tools/analysis')

        env = os.environ.copy()
        env['PLAIDML_EVENTLOG_FILENAME'] = os.path.join(cwd, 'testdata',
                                                        'small_mobilenet_coco_log.gz')

        notebook = os.path.join(cwd, 'Analysis.ipynb')
        conda_env = os.getenv('CONDA_DEFAULT_ENV')
        if os.name == 'nt':
            cmd = [os.path.join(conda_env, 'Scripts', 'jupyter-nbconvert.exe')]
        else:
            # we need to use `python $JUPYTER_PATH` to avoid PermissionDenied errors
            # due to bash's limit of 128 chars for the shebang line
            # see: https://github.com/pypa/pip/issues/1773
            # see: https://www.in-ulm.de/~mascheck/various/shebang/#length
            cmd = ['python', os.path.join(conda_env, 'bin', 'jupyter-nbconvert')]
        cmd += ['--execute', '--ExecutePreprocessor.timeout=60', notebook]
        subprocess.check_call(cmd, env=env)


if __name__ == '__main__':
    unittest.main()
