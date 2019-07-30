import gzip
import os

from base.eventing.file import eventlog_pb2 as epb


def _read_varint(f):
    result = 0
    shift = 0
    while True:
        b = f.read(1)
        if not b:
            return 0
        i = int.from_bytes(b, 'big')
        result |= (i & 0x7f) << shift
        if not i & 0x80:
            return result
        shift += 7
        if shift >= 64:
            return 0  # TODO: Consider raising a status instead.


def read_eventlog(filename):
    """Reads an event log data file.

    Args:
        filename (str): A file containing serialized events.

    Yields:
        A sequence of context_pb2.Event events
    """
    # Hack to allow testing from internal and public bazel
    if not os.path.exists(filename):
        filename = os.path.join("../com_intel_plaidml", filename)
    with gzip.open(filename, 'rb') as f:
        while True:
            rec = epb.Record()
            length = _read_varint(f)
            if not length:
                break
            data = f.read(length)
            if len(data) != length:
                break  # Partial data at EOF
            rec.ParseFromString(data)
            for evt in rec.event:
                yield evt
