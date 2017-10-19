from __future__ import print_function

import os
import tempfile
import unittest
import uuid

import plaidml.exceptions
import plaidml.settings as s

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
        s._setup_for_test()

    def testDefaults(self):
        self.assertEquals(s.config, None)
        self.assertEquals(s.device_ids, [])
        self.assertEquals(s.experimental, False) 
        self.assertEquals(s.session, None)
        self.assertEquals(s.telemetry, False)

    def testSetting(self):
        s.config = 'test'
        s.device_ids = ['1','2']
        s.experimental = True
        s.telemetry = True
        s.session = "123"
        self.assertEquals(s.config, 'test')
        self.assertEquals(s.device_ids, ['1', '2'])
        self.assertEquals(s.experimental, True) 
        self.assertEquals(s.session, "123")
        self.assertEquals(s.telemetry, True)

    def testStartSession(self):
        with self.assertRaises(plaidml.exceptions.PlaidMLError):
            s.start_session()
        s.experimental = True
        s.start_session()
        u = uuid.UUID(s.session)

    def testSettingsFileLoading(self):
        with tempfile.NamedTemporaryFile(delete=False) as val:
            val.write(VALID_CONF)
        with tempfile.NamedTemporaryFile(delete=False) as inv:
            inv.write(INVALID_CONF)
        # Explicit settings files should take precedence
        s._setup_for_test(inv.name, inv.name)
        os.environ['PLAIDML_SETTINGS'] = val.name
        s._load()
        self.assertEquals(s.config, 'tmp')
        self.assertEquals(s.experimental, True)
        self.assertEquals(s.device_ids, ['1', '3', '5'])
        self.assertEquals(s.telemetry, True)
 
        # User config should shadow system config
        s._setup_for_test(val.name, inv.name)
        s._load()
        self.assertEquals(s.experimental, True)
        s._setup_for_test('nottafile', inv.name)
        with self.assertRaises(plaidml.exceptions.OutOfRange):
            s._load()
        os.remove(val.name)
        os.remove(inv.name)
  
    def testSettingsOverridesLoading(self):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(VALID_CONF)
            os.environ['PLAIDML_SETTINGS'] = tf.name
        os.environ['PLAIDML_CONFIG'] = 'other'
        s.telemetry = False
        s._load()
        self.assertEquals(s.config, 'other')
        self.assertEquals(s.telemetry, False)
        os.remove(tf.name)
    
if __name__ == '__main__':
    unittest.main()