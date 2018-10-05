# Copyright 2018 Intel Corporation.

from __future__ import print_function

import glob
import os.path
import plaidml
import plaidml.exceptions
import types
import unittest


class ConfigsTest(unittest.TestCase):

    def validateConfig(self, config_filename):
        print('Validating %s' % (config_filename,))
        ctx = plaidml.Context()
        with open(config_filename) as config_file:
            config = config_file.read()
        try:
            for conf in plaidml.devices(ctx, config):
                print('Found device %s: %s' % (conf.name, conf.description))
        except plaidml.exceptions.NotFound:
            print('No devices found matching %s' % (config_filename,))

    @classmethod
    def add_config_test(cls, config_filename):
        basename = os.path.basename(os.path.splitext(config_filename)[0])
        testname = 'test' + ''.join(x.capitalize() for x in basename.split('_'))

        def test_config(self):
            self.validateConfig(config_filename)

        setattr(cls, testname, types.MethodType(test_config, None, cls))


if __name__ == '__main__':
    for config_filename in glob.glob('testing/*.json'):
        ConfigsTest.add_config_test(config_filename)
    unittest.main()
