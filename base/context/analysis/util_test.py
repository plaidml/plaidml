import google.protobuf.text_format as text_format
import pandas
import pandas.util.testing as pt
import unittest
import uuid

from base.context import context_pb2 as cpb
from base.context.analysis import util as ca


class ReaderTests(unittest.TestCase):
    def test_read_protos(self):
        events = ca.read_eventlog('base/context/analysis/testdata/eventlog.gz')
        ev1 = events.next()
        self.assertIsNotNone(ev1)
        self.assertEqual(ev1.verb, 'Hello, World!')
        with self.assertRaises(StopIteration):
            events.next()


class LoaderTests(unittest.TestCase):
    def base_eventlog_proto(self):
        eventlog = cpb.Event()
        text_format.Merge("""
            instance_uuid: "\\000\\001\\002\\003\\004\\005\\006\\007\\010\\011\\012\\013\\014\\015\\016\\017"
            parent_instance_uuid: "\\100\\101\\102\\103\\104\\105\\106\\107\\110\\111\\112\\113\\114\\115\\116\\117"
            clock_uuid: "\\200\\201\\202\\203\\204\\205\\206\\207\\210\\211\\212\\213\\214\\215\\216\\217"
            verb: "context::test"
            start_time: { seconds:1 nanos:2 }
            end_time: { seconds:3 nanos:4 }
            """,
                          eventlog)
        return eventlog

    def base_dataframe(self):
        return pandas.DataFrame(
            {'instance': [uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e0f}')],
             'parent': [uuid.UUID('{4041424344-4546-4748-494a-4b4c4d4e4f}')],
             'clock': [uuid.UUID('{8081828384-8586-8788-898a-8b8c8d8e8f}')],
             'verb': ['context::test'],
             'start_time': [pandas.Timedelta(seconds=1, nanoseconds=2)],
             'end_time': [pandas.Timedelta(seconds=3, nanoseconds=4)]});

    def test_full_proto_to_dataframe(self):
        eventlog = self.base_eventlog_proto()
        pdf = ca.events_to_dataframe([eventlog]).sort_index(axis=1)
        pt.assert_frame_equal(pdf, self.base_dataframe())

    def test_proto_to_dataframe_missing_parent(self):
        eventlog = self.base_eventlog_proto()
        eventlog.ClearField("parent_instance_uuid")
        pdf = ca.events_to_dataframe([eventlog]).sort_index(axis=1)
        df = self.base_dataframe()
        df.set_value(0, 'parent', None)
        pt.assert_frame_equal(pdf, df)

    def test_proto_to_dataframe_missing_clock(self):
        eventlog = self.base_eventlog_proto()
        eventlog.ClearField("clock_uuid")
        pdf = ca.events_to_dataframe([eventlog]).sort_index(axis=1)
        df = self.base_dataframe()
        df.set_value(0, 'clock', None)
        pt.assert_frame_equal(pdf, df)

    def test_proto_to_dataframe_missing_verb(self):
        eventlog = self.base_eventlog_proto()
        eventlog.ClearField("verb")
        pdf = ca.events_to_dataframe([eventlog]).sort_index(axis=1)
        df = self.base_dataframe()
        df.set_value(0, 'verb', None)
        pt.assert_frame_equal(pdf, df)

    def test_proto_to_dataframe_missing_start_time(self):
        # Note: this test is slightly more complicated, because for some reason
        # pandas infers dtype datetime[ns] instead of timedelta[ns] for a column
        # if the only value in the column is a NAT Timedelta.  To work around this
        # (because this is unlikely to occur with real data), we duplicate
        # the first event entry in the proto before clearing the time field.
        ev1 = self.base_eventlog_proto()
        ev2 = self.base_eventlog_proto()
        ev1.ClearField("start_time")
        pdf = ca.events_to_dataframe([ev1, ev2]).sort_index(axis=1)
        df = self.base_dataframe()
        df = pandas.concat([df, df], ignore_index=True)
        df.set_value(0, 'start_time', pandas.Timedelta('nan'))
        pt.assert_frame_equal(pdf, df)

    def test_proto_to_dataframe_missing_end_time(self):
        # Note: this test is slightly more complicated, because for some reason
        # pandas infers dtype datetime[ns] instead of timedelta[ns] for a column
        # if the only value in the column is a NAT Timedelta.  To work around this
        # (because this is unlikely to occur with real data), we duplicate
        # the first event entry in the proto before clearing the time field.
        ev1 = self.base_eventlog_proto()
        ev2 = self.base_eventlog_proto()
        ev1.ClearField("end_time")
        pdf = ca.events_to_dataframe([ev1, ev2]).sort_index(axis=1)
        df = self.base_dataframe()
        df = pandas.concat([df, df], ignore_index=True)
        df.set_value(0, 'end_time', pandas.Timedelta('nan'))
        pt.assert_frame_equal(pdf, df)


class CookTests(unittest.TestCase):
    def test_join_start_and_end_times(self):
        raw = pandas.DataFrame([
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e0f}'),
             'parent': uuid.UUID('{4041424344-4546-4748-494a-4b4c4d4e4f}'),
             'clock': uuid.UUID('{8081828384-8586-8788-898a-8b8c8d8e8f}'),
             'verb': 'context::test',
             'start_time': pandas.Timedelta(seconds=2),
             'end_time': pandas.Timedelta('nat')},
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e0f}'),
             'parent': uuid.UUID('{4041424344-4546-4748-494a-4b4c4d4e4f}'),
             'clock': uuid.UUID('{8081828384-8586-8788-898a-8b8c8d8e8f}'),
             'verb': 'context::test',
             'start_time': pandas.Timedelta('nat'),
             'end_time': pandas.Timedelta(seconds=4)},
        ]);
        cooked = pandas.DataFrame.from_records([
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e0f}'),
             'parent': uuid.UUID('{4041424344-4546-4748-494a-4b4c4d4e4f}'),
             'verb': 'context::test',
             'start_time': pandas.Timedelta(seconds=0),
             'end_time': pandas.Timedelta(seconds=2)},
        ], index=['instance']);

        pt.assert_frame_equal(ca.cook(raw).sort_index(axis=1), cooked)

    def test_align_two_clocks(self):
        raw = pandas.DataFrame([
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e0f}'),
             'parent': uuid.UUID('{4041424344-4546-4748-494a-4b4c4d4e4f}'),
             'clock': uuid.UUID('{8081828384-8586-8788-898a-8b8c8d8e8f}'),
             'verb': 'context::test',
             'start_time': pandas.Timedelta(seconds=2),
             'end_time': pandas.Timedelta(seconds=4)},
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e10}'),
             'parent': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e0f}'),
             'clock': uuid.UUID('{8081828384-8586-8788-898a-8b8c8d8e90}'),
             'verb': 'context::test::subclock',
             'start_time': pandas.Timedelta(seconds=10),
             'end_time': pandas.Timedelta(seconds=15)},
        ]);
        cooked = pandas.DataFrame.from_records([
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e0f}'),
             'parent': uuid.UUID('{4041424344-4546-4748-494a-4b4c4d4e4f}'),
             'verb': 'context::test',
             'start_time': pandas.Timedelta(seconds=0),
             'end_time': pandas.Timedelta(seconds=2)},
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e10}'),
             'parent': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e0f}'),
             'verb': 'context::test::subclock',
             'start_time': pandas.Timedelta(seconds=0),
             'end_time': pandas.Timedelta(seconds=5)},
        ], index=['instance']);

        pt.assert_frame_equal(ca.cook(raw).sort_index(axis=1), cooked)

    def test_cross_align_three_clocks(self):
        raw = pandas.DataFrame([
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e0f}'),
             'parent': uuid.UUID('{4041424344-4546-4748-494a-4b4c4d4e4f}'),
             'clock': uuid.UUID('{8081828384-8586-8788-898a-8b8c8d8e8f}'),
             'verb': 'context::test',
             'start_time': pandas.Timedelta(seconds=2),
             'end_time': pandas.Timedelta(seconds=4)},
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e10}'),
             'parent': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e0f}'),
             'clock': uuid.UUID('{8081828384-8586-8788-898a-8b8c8d8e90}'),
             'verb': 'context::test::subclock',
             'start_time': pandas.Timedelta(seconds=10),
             'end_time': pandas.Timedelta(seconds=15)},
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e11}'),
             'parent': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e0f}'),
             'clock': uuid.UUID('{8081828384-8586-8788-898a-8b8c8d8e90}'),
             'verb': 'context::test::subclock::second_action',
             'start_time': pandas.Timedelta(seconds=12),
             'end_time': pandas.Timedelta(seconds=14)},
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e12}'),
             'parent': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e11}'),
             'clock': uuid.UUID('{8081828384-8586-8788-898a-8b8c8d8e91}'),
             'verb': 'context::test::subclock::sub-sub-clock',
             'start_time': pandas.Timedelta(seconds=1),
             'end_time': pandas.Timedelta(seconds=9)},
        ]);
        cooked = pandas.DataFrame.from_records([
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e0f}'),
             'parent': uuid.UUID('{4041424344-4546-4748-494a-4b4c4d4e4f}'),
             'verb': 'context::test',
             'start_time': pandas.Timedelta(seconds=0),
             'end_time': pandas.Timedelta(seconds=2)},
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e10}'),
             'parent': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e0f}'),
             'verb': 'context::test::subclock',
             'start_time': pandas.Timedelta(seconds=0),
             'end_time': pandas.Timedelta(seconds=5)},
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e11}'),
             'parent': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e0f}'),
             'verb': 'context::test::subclock::second_action',
             'start_time': pandas.Timedelta(seconds=2),
             'end_time': pandas.Timedelta(seconds=4)},
            {'instance': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e12}'),
             'parent': uuid.UUID('{0001020304-0506-0708-090a-0b0c0d0e11}'),
             'verb': 'context::test::subclock::sub-sub-clock',
             'start_time': pandas.Timedelta(seconds=2),
             'end_time': pandas.Timedelta(seconds=10)},
        ], index=['instance']);

        pt.assert_frame_equal(ca.cook(raw).sort_index(axis=1), cooked)


if __name__ == '__main__':
    unittest.main()
