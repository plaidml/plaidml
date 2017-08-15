# Copyright Vertex.AI.

"""Intelligently displays buffer differences between two trace logs."""

from __future__ import absolute_import, print_function

import argparse
import itertools
import numpy as np
import uuid

from tile.sdk.src.event import event_pb2
from tile.sdk.src.event import filelog


class Differ(object):
    def __init__(self):
        self._step_index = 0
        self._step_max = 0

    def diff(self, lhs, rhs):
        lhs_evt_id = ''
        for (l, r) in itertools.izip(filelog.Read(lhs), filelog.Read(rhs)):
            if lhs_evt_id != l.instance_uuid:
                if l.verb != 'vertexai::keras::Step':
                    continue
                lhs_evt_id = l.instance_uuid
                self._step_index = self._step_index + 1
                if self._step_max and self._step_max < self._step_index:
                    return
            self._diff_buffers(self._yield_buffers(l), self._yield_buffers(r))

    def _yield_buffers(self, evt):
        for md in evt.metadata:
            if md.Is(event_pb2.Buffer.DESCRIPTOR):
                buf = event_pb2.Buffer()
                md.Unpack(buf)
                yield buf

    def _diff_buffers(self, lhs, rhs):
        for (l, r) in itertools.izip(lhs, rhs):
            npl = self._to_ndarray(l)
            npr = self._to_ndarray(r)
            delta = npl - npr
            max_delta = delta.max()
            if max_delta > .01:
                self._print_buffer(l, l.comment)
                print('max delta:', max_delta)
                print('delta:', delta)
            else:
                self._print_buffer(l, l.comment.split(':', 2)[0])

    def _to_ndarray(self, buf):
        out = np.fromstring(buf.data, dtype=buf.dtype)
        out.shape = tuple([i for i in buf.shape])
        return out

    def _print_buffer(self, buf, comment):
        print(' Batch %d dtype=%s shape=(%s) len=%d %s' % (self._step_index, buf.dtype, ', '.join([str(i) for i in buf.shape]), len(buf.data), comment))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lhs')
    parser.add_argument('rhs')
    args = parser.parse_args()
    Differ().diff(args.lhs, args.rhs)
