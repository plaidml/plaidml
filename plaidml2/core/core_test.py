# Copyright 2019 Intel Corporation.

import unittest

import plaidml2 as plaidml
import plaidml2.core as pcore


class TestCore(unittest.TestCase):

    def test_settings(self):
        pcore.settings.set('FOO', 'bar')
        self.assertEqual(pcore.settings.get('FOO'), 'bar')
        settings = pcore.settings.all()
        print(settings)
        self.assertIn('FOO', settings)


if __name__ == '__main__':
    unittest.main()
