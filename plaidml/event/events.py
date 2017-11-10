# Copyright Vertex.AI.
"""Provides common event handling functionality."""

from __future__ import absolute_import, print_function

import contextlib
from datetime import datetime
import uuid

from . import event_pb2


def _Now():
    """Returns the current time, as a timedelta from the system clock's epoch."""
    return datetime.utcnow() - datetime.utcfromtimestamp(0)


class Context(object):
    """An execution context for eventing."""

    def __init__(self, parent=None):
        self.id = uuid.uuid4()
        self.eventlog = parent.eventlog if parent else None
        self.domain = parent.domain if parent else None

    @contextlib.contextmanager
    def Activity(self, verb):
        c = Context(parent=self)
        if c.eventlog:
            with c.eventlog.event() as evt:
                evt.parent_instance_uuid = self.id.bytes
                evt.instance_uuid = c.id.bytes
                if c.domain:
                    evt.domain_uuid = c.domain.bytes
                evt.verb = verb
                evt.start_time.FromTimedelta(_Now())

        yield c

        if c.eventlog:
            with c.eventlog.event() as evt:
                evt.instance_uuid = c.id.bytes
                evt.end_time.FromTimedelta(_Now())


def LogDomain(ctx, creator):
    """Emits a domain event into the context's log.

    Args:
        ctx: The logging context
        creator: The point in the codebase that's creating the domain

    Returns:
        The domain's UUID.
    """
    u = uuid.uuid4()
    if ctx.eventlog:
        with ctx.eventlog.event() as evt:
            evt.instance_uuid = u.bytes
            evt.verb = 'vertexai::Domain'
            dom = event_pb2.Domain()
            dom.creator = creator
            evt.metadata.add().Pack(dom)
            now = _Now()
            evt.start_time.FromTimedelta(now)
            evt.end_time.FromTimedelta(now)
    return u


def LogBufferInfo(ctx, data, comment=None):
    """Adds an event with the supplied buffer data.

    Args:
        ctx: The logging context
        data: The buffer data (as a numpy.ndarray)
        comment: A note to attach to the buffer data
    """
    if not ctx.eventlog:
        return

    with ctx.eventlog.event() as evt:
        evt.instance_uuid = ctx.id.bytes
        b = event_pb2.Buffer()
        if comment:
            b.comment = comment
            b.dtype = data.dtype.name
            b.shape.extend(data.shape)
            b.data = data.tobytes()
            evt.metadata.add().Pack(b)
