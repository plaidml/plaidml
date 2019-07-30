import unittest

from tools.analysis import util


class ReaderTests(unittest.TestCase):

    def test_read_protos(self):
        events = util.read_eventlog('tools/analysis/testdata/eventlog.gz')
        ev1 = next(events)
        self.assertIsNotNone(ev1)
        self.assertEqual(ev1.verb, 'Hello, World!')
        with self.assertRaises(StopIteration):
            next(events)


if __name__ == '__main__':
    unittest.main()
