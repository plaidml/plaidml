# Copyright Vertex.AI.
"""Implements access to Vertex.AI's event log file format."""

from __future__ import absolute_import, print_function

import contextlib

from google.protobuf.internal import encoder as gpbe
from google.protobuf.internal import decoder as gpbd

from . import event_pb2


class Writer(object):
    """Writes events to an event trace file."""

    def __init__(self, f):
        if not hasattr(f, 'write'):
            self._f = open(f, 'w')
        else:
            self._f = f

        rec = event_pb2.FileRecord()
        rec.magic.value = event_pb2.FileMagic.Eventlog
        self._write(rec)

    @contextlib.contextmanager
    def event(self):
        rec = event_pb2.FileRecord()
        yield rec.event.add()
        self._write(rec)

    def _write(self, rec):
        rs = rec.SerializeToString()
        gpbe._EncodeVarint(self._f.write, len(rs))
        self._f.write(rs)

    def close(self):
        self._f.close()


def Read(f):
    """Reads an event trace file.

    Args:
        f: The file to read.

    Yields:
        The events contained in the file.
    """
    close = not hasattr(f, 'read')
    if close:
        f = open(f, 'r')

    try:
        while True:
            pos = f.tell()
            l_data = f.read(10)  # 10 is always enough
            l, l_sz = gpbd._DecodeVarint(l_data, 0)
            f.seek(pos + l_sz)
            data = f.read(l)
            rec = event_pb2.FileRecord()
            rec.ParseFromString(data)
            for evt in rec.event:
                yield evt

    finally:
        if close:
            f.close()
