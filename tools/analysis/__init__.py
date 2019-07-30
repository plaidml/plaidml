# Copyright 2018 Intel Corporation.

import datetime
import logging
import uuid

import defaultlist
import networkx as nx
import numpy as np
import pandas as pd
from google.protobuf import symbol_database

import tile.hal.opencl.opencl_pb2
import tile.platform.local_machine.local_machine_pb2
import tile.proto.hal_pb2
import tile.proto.schedule_pb2
import util

_WELL_KNOWN_METADATA = {
    'ocl_runinfo': (tile.hal.opencl.opencl_pb2.RunInfo,),
    'ocl_kernelinfo': (tile.hal.opencl.opencl_pb2.KernelInfo,),
    'ocl_buildinfo': (tile.hal.opencl.opencl_pb2.BuildInfo,),
    'ocl_deviceinfo': (tile.hal.opencl.opencl_pb2.DeviceInfo,),
    'ocl_platforminfo': (tile.hal.opencl.opencl_pb2.PlatformInfo,),
    'hal_compilationinfo': (tile.proto.hal_pb2.CompilationInfo,),
    'lpt_schedule': (
        tile.proto.schedule_pb2.Schedule,
        tile.platform.local_machine.local_machine_pb2.Schedule,
    )
}

_LOGGER = logging.getLogger(__name__)

VERBS_EXECUTING = frozenset(('tile::hal::opencl::Executing', 'tile::hal::cpu::Executing'))


class Activity(object):
    """Represents a single activity within a stream."""
    __slots__ = [
        'idx', 'parent', 'stream', 'verb', 'clock', '_relative_start_time', '_relative_end_time',
        'metadata', 'children'
    ]

    def __init__(self, stream):
        self.idx = 0  # N.B. Index within the scope, not the stream.  0 == Invalid/unassigned.
        self.parent = None
        self.stream = stream
        self.verb = ''
        self.clock = None
        self._relative_start_time = None
        self._relative_end_time = None
        self.metadata = []
        self.children = set()

    def __str__(self):
        return '{} {}'.format(id(self), self.verb)

    def __repr__(self):
        return 'Activity(id={}, verb={}, start={}, end={}, clock={})'.format(
            id(self), self.verb, datetime.timedelta(seconds=self.start_time),
            datetime.timedelta(seconds=self.end_time), self.clock)

    @property
    def relative_start_time(self):
        if self._relative_start_time is not None:
            return self._relative_start_time
        return 0.

    @relative_start_time.setter
    def relative_start_time(self, value):
        self._relative_start_time = value

    @property
    def relative_end_time(self):
        if self._relative_end_time is not None:
            return self._relative_end_time
        return self.clock.end

    @relative_end_time.setter
    def relative_end_time(self, value):
        self._relative_end_time = value

    @property
    def start_time(self):
        return self.clock.epoch + self.relative_start_time

    @property
    def end_time(self):
        return self.clock.epoch + self.relative_end_time

    @property
    def elapsed_time(self):
        return self.relative_end_time - self.relative_start_time

    def enclosing_metadata(self):
        """Yields the metadata of the activity and its parents."""
        for act in self.path():
            for mdat in act.metadata:
                yield mdat

    def path(self):
        """Yields the activity and its parents."""
        act = self
        while act:
            yield act
            act = act.parent

    def subtree(self):
        """Yields the subtree of activities rooted at the current activity.

        Activities are yielded in depth-first scan, parents before children.
        """
        pending = [self]
        while pending:
            act = pending.pop()
            yield act
            children = list(act.children)
            children.sort(key=lambda a: a.start_time, reverse=True)
            pending.extend(children)

    def _first_metadatum(self, name):
        """Returns the first metadatum that matches a given class.

        Args:
            name: The name of the class the caller is looking for.

        Returns:
            The first instance of the class found in the event's metadata
            or the event's parent's metadata.

        Raises:
            KeyError: There is no metadatum matching the given class.
        """
        for cls in _WELL_KNOWN_METADATA[name]:
            for mdat in self.enclosing_metadata():
                if isinstance(mdat, cls):
                    return mdat
        raise KeyError(name)

    def __getattr__(self, name):
        if name in _WELL_KNOWN_METADATA:
            return self._first_metadatum(name)
        raise AttributeError


class Clock(object):
    """An instance of a clock."""

    def __init__(self, stream):
        self.stream = stream
        self.epoch = 0.
        self.end = 0.

    def __repr__(self):
        return 'Clock(id={}, stream={}, epoch={}, end={})'.format(id(self), self.stream,
                                                                  self.epoch, self.end)


class Stream(object):
    """A stream of activities."""

    def __init__(self, scope, stream_uuid):
        self.scope = scope
        self.uuid = stream_uuid
        self.clocks = defaultlist.defaultlist(lambda: Clock(self))
        self.activities = defaultlist.defaultlist(lambda: Activity(self))
        self.activities[0].verb = '<root>'
        self.activities[0].idx = scope.make_idx()
        self.activities[0].clock = self.clocks[0]


class _DefaultKeyDict(dict):
    """Like defaultdict, but supplies the key to the factory."""

    def __init__(self, factory):
        self.factory = factory

    def __missing__(self, key):
        self[key] = self.factory(key)
        return self[key]


def as_dataframe(scope, activities):
    """Builds a dataframe out of the supplied activities.
    
    Args:
        scope (Scope): The scope these activities came from.
        activities (iterable(Activity)): The activities to use.
    
    Returns:
        pandas.DataFrame: A dataframe containing the activities.
    """
    records = [(act.idx, act.verb, act.start_time, act.end_time, act,
                act.parent.idx if act.parent else 0) for act in activities]
    dtype = [('idx', 'i'), ('verb', np.unicode_, scope._verb_max_len), ('start', 'f8'),
             ('end', 'f8'), ('activity', 'O'), ('parent', 'i')]
    return pd.DataFrame.from_records(np.array(records, dtype=dtype), index='idx')


class Scope(object):
    """A scope for a set of event streams.

    Instances of this class handle event cross-references and event
    clocks.  Events from multiple scopes may belong to the same
    EventSet, but will be treated as though they have completely
    isolated clock domains; parent/child relationships are only
    defined within a particular scope.
    """

    def __init__(self):
        super(Scope, self).__init__()

        # The known streams, by uuid.
        self.streams = _DefaultKeyDict(lambda stream_uuid: Stream(self, uuid.UUID(bytes=stream_uuid
                                                                                 )))

        # The previous scope-wide index assigned to an Activity.
        self._prev_idx = 0

        # The maximum verb length for the activities in this scope.
        self._verb_max_len = 0

    def make_idx(self):
        self._prev_idx += 1
        return self._prev_idx

    def get_activity(self, source, activity_id):
        stream = source.stream
        if activity_id.stream_uuid:
            stream = self.streams[activity_id.stream_uuid]
        return stream.activities[activity_id.index]

    @property
    def activities(self):
        for stream in self.streams.values():
            for act in stream.activities:
                yield act

    def read_eventlog(self, filename):
        """Reads events from an eventlog, and adds them to the scope."""
        symdb = symbol_database.Default()
        last_uuid = None
        for pbevt in util.read_eventlog(filename):
            if not pbevt.activity_id.index:
                continue
            if pbevt.activity_id.stream_uuid or not last_uuid:
                last_uuid = pbevt.activity_id.stream_uuid
                stream = self.streams[last_uuid]
            act = stream.activities[pbevt.activity_id.index]
            if not act.idx:
                act.idx = self.make_idx()
            if pbevt.HasField('parent_id'):
                act.parent = self.get_activity(act, pbevt.parent_id)
                act.parent.children.add(act)
            if pbevt.verb:
                act.verb = pbevt.verb
                self._verb_max_len = max(self._verb_max_len, len(act.verb))
            if pbevt.clock_id:
                act.clock = stream.clocks[pbevt.clock_id.index]
            if pbevt.start_time.seconds or pbevt.start_time.nanos:
                act.relative_start_time = (float(pbevt.start_time.seconds) +
                                           float(pbevt.start_time.nanos) / 1000000000.)
            if pbevt.end_time.seconds or pbevt.end_time.nanos:
                act.relative_end_time = (float(pbevt.end_time.seconds) +
                                         float(pbevt.end_time.nanos) / 1000000000.)
            if act.clock and act.relative_end_time:
                act.clock.end = max(act.clock.end, act.relative_end_time)
            for datum in pbevt.metadata:
                try:
                    (_, sym) = datum.type_url.split('/')
                    msg = symdb.GetSymbol(sym)()
                except:
                    _LOGGER.warning('Unable to lookup metadata type "%(type_url)s"',
                                    type_url=datum.type_url)
                    continue
                try:
                    datum.Unpack(msg)
                except:
                    _LOGGER.warning('Unable to unpack metadata type "%(type_url)s"',
                                    type_url=datum.type_url)
                    continue
                act.metadata.append(msg)

        self._adjust_clocks()

    def _adjust_clocks(self):
        """Aligns the clocks within the scope.

        This method uses activity parent/child relationships that cross
        clock domains to determine the minimum clock skews required to
        make child events start after their parents.

        Clocks that do not have parents are assumed to produce event
        times relative to the Unix epoch, 1970-01-01T00:00:00Z.
        """

        # We start by assuming that all clocks run at the same speed, just to make
        # things simple.  (It's possible to perform alignment when the clocks are
        # running at different speeds, but it's a bit more complex and has some
        # interesting issues to think about, so we're not doing it for now).
        #
        # Because they're running at the same speed, when two clocks are causally
        # related, we can determine the minimum delta that must be applied to cause
        # the happens-after event to occur after the happens-before event.  So we can
        # take all cross-clock parent/child event pairs, and compute the maximum of
        # the minimum deltas; this is the delta we need for that clock pair.
        #
        # We arrange the clocks in a directed graph, where the nodes represent the
        # clocks and the edges are weighted by the deltas between pairs of clocks.
        #
        # For each connected subgraph, we pick a node at random, and label it zero.
        # We then walk an arbitrary spanning tree from that node, adding the weights
        # of the traversed edges as we go and labeling the nodes with the cumulative
        # weight from the initial node.
        #
        # We then find the minimum node for each subgraph, and subtract its label
        # from the labels of every other node in the subgraph.  That gets us a set of
        # causally related clock skews; we then add each clock's label to the start
        # and end times of all events measured relative to that clock, and we end up
        # with a relatively aligned dataset.
        graph = nx.DiGraph()
        for stream in self.streams.values():
            for clock in stream.clocks:
                clock.epoch = 0.
            graph.add_nodes_from(stream.clocks)
            for act in stream.activities:
                if not act.parent or act.parent.clock == act.clock:
                    continue
                min_delta = act.parent.relative_start_time - act.relative_start_time
                if min_delta < 0.:
                    continue
                if graph.has_edge(act.parent.clock, act.clock):
                    prev_delta = graph[act.parent.clock][act.clock]['delta']
                    if prev_delta < min_delta:
                        graph[act.parent.clock][act.clock]['delta'] = min_delta
                else:
                    graph.add_edge(act.parent.clock, act.clock, delta=min_delta)
        for subgraph in (graph.subgraph(c) for c in nx.weakly_connected_components(graph)):
            min_epoch = 0.
            for start in subgraph:
                break
            seen = set([start])
            pending = list([start])
            while pending:
                clock = pending.pop()
                for neighbor in subgraph.successors(clock):
                    if neighbor in seen:
                        continue
                    neighbor.epoch = clock.epoch + subgraph[clock][neighbor]['delta']
                    seen.add(neighbor)
                    pending.append(neighbor)
                for neighbor in subgraph.predecessors(clock):
                    if neighbor in seen:
                        continue
                    neighbor.epoch = clock.epoch - subgraph[neighbor][clock]['delta']
                    if neighbor.epoch < min_epoch:
                        min_epoch = neighbor.epoch
                    seen.add(neighbor)
                    pending.append(neighbor)
            if min_epoch < 0.:
                for clock in subgraph:
                    clock.epoch -= min_epoch


def connections(activities):
    """Enumerates the connections within a list of activities.

    Args:
      activities: The activities to process.  Parents/children outside these are ignored.

    Returns:
      A list of two-tuples.  Each tuple is a (parent, child) relationship.
    """
    acts = frozenset(activities)
    cons = []
    for parent in acts:
        for child in parent.children:
            if child in acts:
                cons.append((parent, child))
    return cons
