# Copyright 2019 Intel Corporation.

import unittest

import plaidml as plaidml
import plaidml.core as pcore
from plaidml.ffi import lib


class TestCore(unittest.TestCase):

    def test_settings(self):
        pcore.settings.set('FOO', 'bar')
        self.assertEqual(pcore.settings.get('FOO'), 'bar')
        settings = pcore.settings.all()
        print(settings)
        self.assertIn('FOO', settings)

    def test_get_strs(self):
        devices = pcore.get_strs(lib.plaidml_targets_get)
        self.assertIn('llvm_cpu', devices)


if __name__ == '__main__':
    unittest.main()
