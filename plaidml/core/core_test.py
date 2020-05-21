# Copyright 2019 Intel Corporation.

import unittest

import plaidml as plaidml
import plaidml.core as pcore
import plaidml.edsl
from plaidml.ffi import lib


class TestCore(unittest.TestCase):

    def test_settings(self):
        pcore.settings.set('FOO', 'bar')
        self.assertEqual(pcore.settings.get('FOO'), 'bar')
        settings = pcore.settings.all()
        print('settings:', settings)
        self.assertIn('FOO', settings)

    def test_get_strings(self):
        targets = pcore.list_targets()
        print('targets:', targets)
        self.assertIn('llvm_cpu', targets)


if __name__ == '__main__':
    unittest.main()
