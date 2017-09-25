import gzip
import numpy as np
import numpy.matlib as matlib
import os
import pandas as pd
import uuid

from base.eventing.file import eventlog_pb2 as epb
from base.context import context_pb2 as cpb


def _read_varint(f):
    result = 0
    shift = 0
    while True:
        s = f.read(1)
        if s == '':
            return 0
        b = bytearray()
        b.extend(s)
        result |= (b[0] & 0x7f) << shift
        if not b[0] & 0x80:
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
        filename = os.path.join("../vertexai_plaidml",  filename)
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


def events_to_dataframe(events):
    """Transforms a sequence of context_pb2.Event data into a pandas.DataFrame.

    Args:
        events (context_pb2.Event[]): A sequence of event protocol buffers.

    Returns:
        pandas.DataFrame: The event data.
    """
    def to_rec(evt):
        """A helper function to convert one event record to an event tuple."""
        return (uuid.UUID(bytes=evt.instance_uuid),
                uuid.UUID(bytes=evt.parent_instance_uuid)
                            if evt.parent_instance_uuid else None,
                evt.verb if evt.verb else None,
                uuid.UUID(bytes=evt.clock_uuid) if evt.clock_uuid else None,
                pd.Timedelta(seconds=evt.start_time.seconds,
                             nanoseconds=evt.start_time.nanos)
                            if evt.HasField('start_time') else pd.Timedelta('nan'),
                pd.Timedelta(seconds=evt.end_time.seconds,
                             nanoseconds=evt.end_time.nanos)
                            if evt.HasField('end_time') else pd.Timedelta('nan'))

    return pd.DataFrame.from_records([to_rec(evt) for evt in events],
                                     columns=['instance', 'parent', 'verb',
                                              'clock', 'start_time',
                                              'end_time'])


def cook(df):
    """Various rewrites of raw event log data to make it easier to grok.

    The most important rewrite is that this routine adjusts clock skews:
    events come from multiple clock domains, and to make sense of them,
    we need to align them s.t. causally connected events have timestamps
    later than their causes.

    Additionally, this routine combines event-start and event-end events,
    and sorts the dataset by start time.

    Args:
        df (pandas.DataFrame): The raw event log data.

    Returns:
        pandas.DataFrame: The cooked event log data.
    """
    # Combine per-instance data records.
    df = df.groupby('instance', as_index=False, sort=False).fillna(method='backfill')
    df.dropna(inplace=True)
    df = df.groupby('instance', as_index=False, sort=False).first()

    # Use the instance ID as the index.
    df.set_index('instance', inplace=True)

    # Align timestamps for each clock to zero.
    def sub_min_ts(x):
        min_start = pd.Timedelta(x['start_time'].min())
        x['start_time'] = x['start_time'] - min_start
        x['end_time'] = x['end_time'] - min_start
        return x
    df = df.groupby('clock', as_index=False, sort=False).apply(sub_min_ts)

    # Align clocks, using record parent references.
    #
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
    #
    # For the representation, we use a simple NxN adjacency matrix, which is
    # likely to work for as many individual clocks as we're likely to see.

    # To start: build a dataframe in which each row is augmented with its
    # parent's data.
    aug = df.merge(df, how='left', left_on='parent', right_index=True,
                   suffixes=('', '_parent'))

    # Compute clock_delta: the time to add to a clock to make it occur just after
    # its parent's clock.  This also lets us use 0 as a "not connected" sentinel
    # later on.
    nano_1 = pd.Timedelta(nanoseconds=1)

    def compute_clock_delta(x):
        try:
            if pd.notnull(x['clock_parent']) and x['clock_parent'] != x['clock']:
                return x['start_time_parent'] - x['start_time'] + nano_1
        except KeyError:
            pass
        return pd.Timedelta('nan')

    aug['clock_delta'] = aug.apply(compute_clock_delta, axis=1)

    # Drop all rows containing NaT clock deltas -- they're uninteresting.
    aug = aug.dropna(subset=['clock_delta'])

    # For each pair of clocks, extract the maximum observed delta.  After this,
    # 'deltas' is a pd.Series indexed by (clock, clock_parent), containing that
    # maximum observed delta.
    deltagroup = aug.groupby(['clock', 'clock_parent'], sort=False)
    deltas = deltagroup.apply(lambda x: x['clock_delta'].max())

    # Now, how many clocks are there?  Construct a dictionary from clocks to
    # unique 0-based indicies that we can use as matrix indicies.
    clocks = {}
    for (cidx, clk) in enumerate(df['clock'].drop_duplicates()):
        clocks[clk] = cidx

    # Build the delta matrix,
    # where dm[row,col] is the delta to add to clock 'row' to get to clock 'col'.
    # 0 == These two clocks are not directly connected.
    dm = matlib.zeros((len(clocks), len(clocks)), dtype=np.dtype('timedelta64[ns]'))
    for ((clock, parent), offset) in deltas.iteritems():
        dm.itemset((clocks[clock], clocks[parent]), -offset)
        dm.itemset((clocks[parent], clocks[clock]), offset)

    # Build the per-clock adjustment list
    adjust = [pd.Timedelta(0)] * len(clocks)

    # Now use the graph defined by the delta matrix to fill in the per-clock adjustment list.
    to_be_visited = set(range(len(clocks)))
    while to_be_visited:
        # Pick an element, initialize the per-connected-subset state
        initial_clk = list(to_be_visited)[0]
        connected_visit_pending = [initial_clk]
        to_be_visited.remove(initial_clk)
        connected_subset = set()
        min_so_far = pd.Timedelta(0)

        while connected_visit_pending:
            src = connected_visit_pending.pop()
            connected_subset.add(src)
            adjust_so_far = adjust[src]
            for dest in range(len(clocks)):
                dist = dm.item((src, dest))
                if not dist or dest not in to_be_visited:
                    continue
                # We haven't seen this destination clock before, but we can reach it
                # from this src.
                to_be_visited.remove(dest)
                connected_visit_pending.append(dest)
                dest_adjust = adjust_so_far + pd.Timedelta(dist)
                adjust[dest] = dest_adjust
                if dest_adjust < min_so_far:
                    min_so_far = dest_adjust

    # Align all adjustments in the connected subset such that the minimum
    # adjustment is zero.
    for clk in connected_subset:
        adjust[clk] = adjust[clk] - min_so_far

    # And then apply the adjustments to the dataframe.
    def adjust_clocks(x):
        delta = adjust[clocks[x['clock']]]
        x['start_time'] = x['start_time'] + delta
        x['end_time'] = x['end_time'] + delta
        return x

    df = df.apply(adjust_clocks, axis=1)

    # Now that the adjustments are in place, discard the clocks and re-sort the
    # frame by start time.
    df.drop('clock', axis=1, inplace=True)
    df.sort_values(['start_time'], inplace=True)

    return df


def select_roots(df):
    """Filters a dataset, selecting the root events.

    Args:
        df (pandas.DataFrame): Cooked event data to filter.

    Returns:
        pandas.DataFrame: The filtered event data.  All parent IDs will be zero.
    """
    return df[df['parent'].isin([uuid.UUID(int=0)])]


def select_verbs(df, *verbs):
    """Filters a dataset, selecting events with the given verbs.

    Args:
        df (pandas.DataFrame): Cooked event data to filter.
        verbs (string[]): The verbs to select

    Returns:
        pandas.DataFrame: The filtered event data.
    """
    return df[df['verb'].isin(verbs)]


def select_transitive(df, srcs):
    """Filters a dataset, selecting events derived from events in srcs.

    Args:
        df (pandas.DataFrame): Cooked event data to filter.
        srcs (pandas.DataFrame): The source events to filter for.

    Returns:
        pandas.DataFrame: The filtered event data.
    """
    srcs = srcs.copy(False)
    srcs['source'] = srcs.index
    results = [srcs]

    while len(srcs):
        sel = df[df['parent'].isin(set(srcs.index))]
        srcs = sel.join(sel.merge(srcs, how='left', left_on=['parent'], right_index=True)['source'])
        results.append(srcs)

    return pd.concat(results)
