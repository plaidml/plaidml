from __future__ import print_function

import os
import tempfile
import unittest
import uuid

import plaidml.exceptions
from plaidml import settings

VALID_CONF = r'''{
    "PLAIDML_CONFIG": "tmp",
    "PLAIDML_CONFIG_FILE": "/tmp",
    "PLAIDML_DEVICE_IDS": ["1", "3", "5"],
    "PLAIDML_EXPERIMENTAL": true,
    "PLAIDML_TELEMETRY": true
}
'''
INVALID_CONF = '{"PLAIDML_INVALID":"1"}'


class TestSettings(unittest.TestCase):

    def setUp(self):
        settings._setup_for_test()

    def testDefaults(self):
        self.assertEquals(settings.config, None)
        self.assertEquals(settings.device_ids, [])
        self.assertEquals(settings.experimental, False)
        self.assertEquals(settings.session, None)

    def testSetting(self):
        settings.config = 'test'
        settings.device_ids = ['1', '2']
        settings.experimental = True
        settings.session = "123"
        self.assertEquals(settings.config, 'test')
        self.assertEquals(settings.device_ids, ['1', '2'])
        self.assertEquals(settings.experimental, True)
        self.assertEquals(settings.session, "123")

    def testStartSession(self):
        with self.assertRaises(plaidml.exceptions.PlaidMLError):
            settings.start_session()
        settings.setup = True
        settings.start_session()
        settings._setup_for_test()
        settings.setup = False
        with self.assertRaises(plaidml.exceptions.PlaidMLError):
            settings.start_session()
        settings._setup_for_test()
        settings.experimental = True
        settings.start_session()
        u = uuid.UUID(settings.session)

    def testSettingsFileLoading(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as val:
            val.write(VALID_CONF)
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as inv:
            inv.write(INVALID_CONF)
        # Explicit settings files should take precedence
        settings._setup_for_test(inv.name, inv.name)
        os.environ['PLAIDML_SETTINGS'] = val.name
        settings._load()
        self.assertEquals(settings.config, 'tmp')
        self.assertEquals(settings.experimental, True)
        self.assertEquals(settings.device_ids, ['1', '3', '5'])

        # User config should shadow system config
        settings._setup_for_test(val.name, inv.name)
        settings._load()
        self.assertEquals(settings.experimental, True)
        settings._setup_for_test('nottafile', inv.name)
        with self.assertRaises(plaidml.exceptions.OutOfRange):
            settings._load()
        os.remove(val.name)
        os.remove(inv.name)

    def testSettingsOverridesLoading(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tf:
            tf.write(VALID_CONF)
            os.environ['PLAIDML_SETTINGS'] = tf.name
        os.environ['PLAIDML_CONFIG'] = 'other'
        settings._load()
        self.assertEquals(settings.config, 'other')
        os.remove(tf.name)


if __name__ == '__main__':
    unittest.main()
