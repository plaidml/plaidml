import argparse
import pandas as pd

import base.context.analysis.util as ca


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert Vertex.AI event data to CSV')
  parser.add_argument('-o', '--out', help='The CSV file to write', required=True)
  parser.add_argument('file', help='a binary profile data file')

  args = parser.parse_args()
  ca.cook(ca.events_to_dataframe(ca.read_eventlog(args.file))).to_csv(args.out)
