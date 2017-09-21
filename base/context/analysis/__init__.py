# Copyright Vertex.AI.
from __future__ import division, print_function

import datetime
import logging
import uuid

import networkx as nx

from google.protobuf import symbol_database

import base.context.analysis.util as util

import tile.hal.opencl.opencl_pb2
import tile.hal.cpu.cpu_pb2
import tile.platform.local_machine.local_machine_pb2
import tile.proto.hal_pb2


_WELL_KNOWN_METADATA = {
    'ocl_runinfo': tile.hal.opencl.opencl_pb2.RunInfo,
    'ocl_kernelinfo': tile.hal.opencl.opencl_pb2.KernelInfo,
    'ocl_buildinfo': tile.hal.opencl.opencl_pb2.BuildInfo,
    'ocl_deviceinfo': tile.hal.opencl.opencl_pb2.DeviceInfo,
    'ocl_platforminfo': tile.hal.opencl.opencl_pb2.PlatformInfo,
    'hal_compilationinfo': tile.proto.hal_pb2.CompilationInfo
}

_LOGGER = logging.getLogger(__name__)


class Event(object):
    """Represents a single event."""
    __slots__ = ['parent', 'uuid', 'verb', 'clock', 'relative_start_time', 'relative_end_time',
                 'metadata', 'children']

    def __init__(self, event_uuid):
        self.uuid = event_uuid
        self.metadata = []
        self.children = set()
        self.clock = None
        self.parent = None
        self.relative_start_time = 0.
        self.relative_end_time = 0.
        self.verb = ''

    def __str__(self):
        return '{} {}'.format(self.uuid, self.verb)

    def __repr__(self):
        return 'Event(uuid={}, verb={}, start={}, end={}, clock={})'.format(
            self.uuid, self.verb,
            datetime.timedelta(seconds=self.start_time),
            datetime.timedelta(seconds=self.end_time),
            self.clock)

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
        """Returns a list of the metadata of the event and parents."""
        mdl = []
        evt = self
        while evt:
            mdl.extend(evt.metadata)
            evt = evt.parent
        return mdl

    def _first_metadatum(self, cls):
        """Returns the first metadatum that matches a given class.

        Args:
            cls: The class the caller is looking for.

        Returns:
            The first instance of the class found in the event's metadata
            or the event's parent's metadata.

        Raises:
            KeyError: There is no metadatum matching the given class.
        """
        for md in self.enclosing_metadata():
            if isinstance(md, cls):
                return md
        else:
            raise KeyError

    def __getattr__(self, name):
        if name in _WELL_KNOWN_METADATA:
            return self._first_metadatum(_WELL_KNOWN_METADATA[name])
        raise AttributeError


class Clock(object):
    """An instance of a clock."""

    def __init__(self, clock_uuid):
        self.uuid = clock_uuid
        self.epoch = 0.

    def __repr__(self):
        return 'Clock(id={}, uuid={}, epoch={})'.format(id(self), uuid.UUID(bytes=self.uuid), self.epoch)

class Scope(object):
    """A scope for a set of events.

    Instances of this class handle event cross-references and event
    clocks.  Events from multiple scopes may belong to the same
    EventSet, but will be treated as though they have completely
    isolated clock domains; parent/child relationships are only
    defined within a particular scope.
    """
    def __init__(self):
        super(Scope, self).__init__()

        # The known events and clocks, keyed by uuid.
        self.events = {}
        self._clocks = {}

    def _get_event(self, event_uuid):
        if event_uuid not in self.events:
            evt = Event(uuid.UUID(bytes=event_uuid))
            self.events[event_uuid] = evt
        return self.events[event_uuid]

    def _get_clock(self, clock_uuid):
        if clock_uuid not in self._clocks:
            self._clocks[clock_uuid] = Clock(clock_uuid)
        return self._clocks[clock_uuid]

    def read_eventlog(self, filename):
        """Reads events from an eventlog, and adds them to the scope."""
        symdb = symbol_database.Default()
        for pbevt in util.read_eventlog(filename):
            evt = self._get_event(pbevt.instance_uuid)
            if pbevt.parent_instance_uuid:
                evt.parent = self._get_event(pbevt.parent_instance_uuid)
                evt.parent.children.add(evt)
            if pbevt.verb:
                evt.verb = pbevt.verb
            if pbevt.clock_uuid:
                evt.clock = self._get_clock(pbevt.clock_uuid)
            if pbevt.start_time.seconds or pbevt.start_time.nanos:
                evt.relative_start_time = (float(pbevt.start_time.seconds) +
                                           float(pbevt.start_time.nanos) / 1000000000.)
            if pbevt.end_time.seconds or pbevt.end_time.nanos:
                evt.relative_end_time = (float(pbevt.end_time.seconds) +
                                         float(pbevt.end_time.nanos) / 1000000000.)
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
                evt.metadata.append(msg)

        self._adjust_clocks()


    def _adjust_clocks(self):
        """Aligns the clocks within the event set.

        This method uses event parent/child relationships that cross
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
        for clock in self._clocks.itervalues():
            clock.epoch = 0.
        graph = nx.DiGraph()
        graph.add_nodes_from(self._clocks.iterkeys())
        for evt in self.events.itervalues():
            if not evt.parent or evt.parent.clock == evt.clock:
                continue
            min_delta = evt.parent.relative_start_time - evt.relative_start_time
            if min_delta < 0.:
                continue
            if graph.has_edge(evt.parent.clock.uuid, evt.clock.uuid):
                prev_delta = graph[evt.parent.clock.uuid][evt.clock.uuid]['delta']
                if prev_delta < min_delta:
                    graph[evt.parent.clock.uuid][evt.clock.uuid]['delta'] = min_delta
            else:
                graph.add_edge(evt.parent.clock.uuid, evt.clock.uuid, {'delta': min_delta})
        for subgraph in nx.weakly_connected_component_subgraphs(graph):
            min_epoch = 0.
            start = subgraph.nodes()[0]
            seen = set([start])
            pending = list([start])
            while pending:
                cuuid = pending.pop()
                for neighbor in subgraph.successors_iter(cuuid):
                    if neighbor in seen:
                        continue
                    self._clocks[neighbor].epoch = (self._clocks[cuuid].epoch +
                                                    subgraph[cuuid][neighbor]['delta'])
                    seen.add(neighbor)
                    pending.append(neighbor)
                for neighbor in subgraph.predecessors_iter(cuuid):
                    if neighbor in seen:
                        continue
                    self._clocks[neighbor].epoch = (self._clocks[cuuid].epoch -
                                                    subgraph[neighbor][cuuid]['delta'])
                    if self._clocks[neighbor].epoch < min_epoch:
                        min_epoch = self._clocks[neighbor].epoch
                    seen.add(neighbor)
                    pending.append(neighbor)
            if min_epoch < 0.:
                for cuuid in subgraph:
                    self._clocks[cuuid].epoch -= min_epoch


def connections(events):
    """Enumerates the connections within a list of events.

    Args:
      events: The events to process.  Parents/children outside these are ignored.

    Returns:
      A list of two-tuples.  Each tuple is a (parent, child) relationship.
    """
    evts = frozenset(events)
    cons = []
    for parent in evts:
        for child in parent.children:
            if child in evts:
                cons.append((parent, child))
    return cons
