from __future__ import print_function

import argparse
import pandas as pd
import sys

import base.context.analysis.util as ca


class walker(object):
  def __init__(self, out):
    self._idx = 1
    self._out = out

  def _next_vertex(self):
    idx = self._idx
    self._idx = idx+1
    return 'v' + str(idx)

  def _walk_srcs(self, events, source_events, this_vertex):
    groups = source_events.groupby('verb')
    total_ns = 0
    for name, child_event_group in groups:
      group_vertex, group_ns = self.walk_group(events, name, child_event_group)
      print('  {} -> {}'.format(this_vertex, group_vertex), file=self._out)
      total_ns = total_ns + group_ns
    return total_ns

  def walk_group(self, events, name, members):
    this_vertex = self._next_vertex()
    count = len(members)
    self_ns = members.time.sum()
    child_ns = self._walk_srcs(events, events[events['parent'].isin(set(members.index))], this_vertex)
    print('  {} [label=\"{}\\n{:,} calls\\nTotal: {:,}ns ({:,}ns children)\\nPerCall: {:,}ns ({:,}ns children)\"]'
          .format(this_vertex, name, count, self_ns, child_ns, self_ns/count, child_ns/count),
          file=self._out)
    return this_vertex, self_ns

  def walk_srcs(self, events, source_events):
    this_vertex = self._next_vertex()
    self._walk_srcs(events, source_events, this_vertex)
    print('  {} [label=\"[EventLog]\"]'.format(this_vertex), file=self._out)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert Vertex.AI event data to GraphViz')
  parser.add_argument('-o', '--out', nargs='?', type=argparse.FileType('w'), help='The GraphViz file to write', default=sys.stdout)
  parser.add_argument('-r', '--root', help='The root verb to trace')
  parser.add_argument('file', help='A binary profile data file')

  args = parser.parse_args()
  events = ca.cook(ca.events_to_dataframe(ca.read_eventlog(args.file)))
  events['time'] = (events['end_time'] - events['start_time']).astype('int64')

  w = walker(args.out)
  print('digraph {', file=args.out)

  if args.root:
    w.walk_group(events, args.root, ca.select_verbs(events, args.root))
  else:
    w.walk_srcs(events, ca.select_roots(events))

  print('}', file=args.out)
