import argparse
import collections
import numpy as np
import pandas as pd

import context.analysis.util as ca


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Generate vertexai profile data summary statistics')
  parser.add_argument('file', help='a binary profile data file')
  parser.add_argument('-r', '--root', help='The root verb to summarize over')

  args = parser.parse_args()
  events = ca.cook(ca.events_to_dataframe(ca.read_eventlog(args.file)))
  if args.root:
    events = ca.select_transitive(events, ca.select_verbs(events, args.root))
  events['time'] = (events['end_time'] - events['start_time']).astype('int64')
  np.set_printoptions(linewidth=120)
  print events.groupby('verb').agg(
    collections.OrderedDict([
      ('verb', 'count'),
      ('time', collections.OrderedDict([
        ('mean', np.mean),
        ('std', np.std),
        ('min', np.min),
        ('max', np.max)]))])).to_string()
