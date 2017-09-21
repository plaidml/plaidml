# Copyright Vertex.AI.

from __future__ import print_function

import logging
import os
import unittest

import nbformat
import runipy.notebook_runner


class AnalysisTest(unittest.TestCase):
    def testRunNotebook(self):
        os.environ['PLAIDML_EVENTLOG_FILENAME'] = os.path.join('base', 'context', 'analysis', 'testdata', 'mnist_mlp_log.gz')

        # N.B. As of 2017-09-07 (when this code was written), runipy only supported up to version 3.
        notebook = nbformat.read(os.path.join('base', 'context', 'analysis', 'Analysis.ipynb'), as_version=3)

        r = runipy.notebook_runner.NotebookRunner(notebook)
        r.run_notebook()


if __name__ == '__main__':
    unittest.main()
